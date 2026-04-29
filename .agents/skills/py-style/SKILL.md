---
name: py-style
description: Enforce and review Python coding style for this project. PEP 8 is the baseline, with the project's explicit rules applied on top. Use this skill whenever you are writing new Python code, reviewing existing Python files, or responding to feedback about code quality. Trigger on any task that involves creating or modifying .py files, reviewing a function or class, or when the user asks about code style, best practices, or refactoring. Always apply these rules proactively — don't wait for the user to ask.
---

# Python Style Guide

PEP 8 is the baseline style guide for all Python code in this project. On top of PEP 8, this skill defines and enforces the project's fourteen explicit coding standards. Apply both layers proactively when writing Python code, and surface every violation when reviewing code.

When PEP 8 and a project-specific rule overlap, follow the stricter project-specific rule. When PEP 8 covers an issue not mentioned here, follow PEP 8.

The rules exist to keep code readable, debuggable, and honest about failure. Clever workarounds that hide bugs are more expensive than crashes that reveal them immediately.

---

## Rule 1 — Imports at the top of the module

All `import` and `from … import` statements belong at the top of the file, grouped after the module docstring. Inline imports (inside functions, methods, or class bodies) obscure dependencies and slow down readers who need to understand what the module requires.

**The only exception:** test code — `test_main()` bodies, pytest fixtures/functions, and `if __name__ == "__main__"` blocks — may use inline imports when they need to pull in heavy or optional dependencies without affecting the module's public load cost.

**Wrong:**
```python
async def save_memory(doc):
    from datetime import datetime  # hidden dependency
    ...
```

**Right:**
```python
from datetime import datetime  # top of file, visible to all readers

async def save_memory(doc):
    ...
```

When reviewing: scan every function and method body. Flag any `import` statement that is not inside test/harness code.

---

## Rule 2 — `try` blocks contain only the risky statement(s)

A `try` block answers the question "what specifically might fail here?" Its scope must be limited to exactly that. Logging, assignment to local variables, calling unrelated functions — none of these belong inside `try`. Keeping the block narrow prevents accidentally catching exceptions from the wrong place, which creates silent, hard-to-diagnose bugs.

**Wrong:**
```python
try:
    result = parse_llm_output(response)
    logger.info(f"Parsed {len(result)} facts")    # cannot raise ParseError
    enriched = enrich(result)                      # unrelated concern
except ParseError:
    result = {}
```

**Right:**
```python
try:
    result = parse_llm_output(response)
except ParseError:
    result = {}
logger.info(f"Parsed {len(result)} facts")
enriched = enrich(result)
```

When reviewing: for each `try` block, verify every statement inside is directly capable of raising the caught exception. Move anything else outside.

---

## Rule 3 — `except` always names a specific exception type

Bare `except:` and `except Exception:` hide the nature of the failure. They catch `KeyboardInterrupt`, `SystemExit`, memory errors, and every other Python exception — usually not what was intended. Naming the exact exception type is a contract: it says "I expect this specific failure, and I know why."

Broad catches are only acceptable at genuine process boundaries (e.g. a top-level Discord event handler or a service main loop) where crashing the entire process would be worse than logging and continuing. Those sites must log with `logger.exception(...)` to preserve the full traceback.

**Wrong:**
```python
try:
    value = int(user_input)
except:             # what error? why?
    value = 0
```

**Wrong:**
```python
except Exception:   # too broad for application logic
    ...
```

**Right:**
```python
try:
    value = int(user_input)
except ValueError:  # specific: int() raises ValueError for bad input
    value = 0
```

When reviewing: flag every `except Exception` and bare `except`. Accept them only at genuine process boundaries that log `logger.exception(...)`.

---

## Rule 4 — Every non-trivial function has a docstring with purpose, args, and return value

Type annotations tell the interpreter what types are expected. Docstrings tell humans *why* a function exists, *what* each argument means conceptually, and *what* the return value represents. A reader should be able to understand a function's contract without reading its body.

Skip the docstring only for one-liner helpers whose name already makes the contract self-evident (e.g. a property that returns a stored attribute).

**Wrong:**
```python
async def update_affinity(global_user_id: str, delta: int) -> int:
    ...
```

**Wrong (too thin):**
```python
async def update_affinity(global_user_id: str, delta: int) -> int:
    """Update affinity."""
    ...
```

**Right:**
```python
async def update_affinity(global_user_id: str, delta: int) -> int:
    """Apply a signed delta to the user's affinity score, clamped to 0–1000.

    Args:
        global_user_id: Internal UUID identifying the user in user_profiles.
        delta: Signed integer; positive values increase affinity, negative decrease.
            The caller is responsible for scaling the raw delta before passing it.

    Returns:
        The new affinity value after clamping.
    """
    ...
```

When reviewing: check every public function and method (those not prefixed with `_` or trivially obvious). Flag missing docstrings and docstrings missing Args or Returns sections.

---

## Rule 5 — Default values live in one place; `.get(key, default)` is not a general substitute

When the same default value appears at every call site via `.get(key, fallback)`, two problems arise: the default can silently diverge across sites, and the code masks the fact that the data was never written in the first place. The right approach is to assign defaults at the point of definition — the config constant, the DB write, the TypedDict — so that the read site can use plain indexing and crash immediately if the value is absent. A crash at the read site means the bug is upstream; surfacing it is the goal.

`.get(key, fallback)` *is* appropriate when reading from genuinely external or untrusted sources: LLM JSON output, API responses, user-provided dicts, or config keys that are intentionally optional.

**Wrong — scattered defaults:**
```python
affinity = doc.get("affinity", 500)       # site 1
affinity = state.get("affinity", 500)     # site 2 — what if it diverges?
affinity = profile.get("affinity", 500)   # site 3
```

**Right — one source of truth:**
```python
# config.py
AFFINITY_DEFAULT = 500

# db.py — the write sets the default once
new_profile = {"affinity": AFFINITY_DEFAULT, ...}
await db.user_profiles.insert_one(new_profile)

# everywhere else — plain indexing; crash if missing signals a write bug
affinity = doc["affinity"]
```

**Acceptable `.get()` uses:**
```python
# LLM output is external and may be incomplete
depth = parsed_llm_output.get("depth", "DEEP")

# Genuinely optional config key
debug = config.get("DEBUG_MODE", False)
```

When reviewing: for each `.get(key, value)` call on an internal dict, ask whether the fallback papers over a missing upstream write. If yes, flag it.

---

## Rule 6 — `try-except` is not default error handling; crashes are informative

If internal code crashes, that crash is a bug report — it tells you exactly where the assumption was wrong. Wrapping internal logic in `try-except` silences the report and delays the diagnosis. Code is *expected* to crash when called incorrectly; the crash surfaces the real bug.

`try-except` is justified only when the failure source is genuinely outside the codebase: LLM output (which is structurally unpredictable), network I/O (where transient failures are normal), external APIs, file system operations on user-supplied paths, and OS-level calls.

**Wrong — suppressing internal bugs:**
```python
try:
    result = compute_embedding(text)
except Exception:
    result = []          # embedding service is expected to work; this hides failures
    logger.warning("embedding failed")
```

**Wrong — using try-except to handle a missing key:**
```python
try:
    user_id = state["global_user_id"]
except KeyError:
    user_id = "anonymous"   # fix the caller instead; the key should always be there
```

**Wrong — defensive DB wrap:**
```python
try:
    await db.user_profiles.update_one(...)
except Exception:
    pass    # DB is internal; failures should propagate
```

**Right — wrapping only genuine external uncertainty:**
```python
# LLM output is external; malformed JSON is a realistic and expected outcome
try:
    parsed = json.loads(llm_response)
except json.JSONDecodeError as exc:
    logger.exception(f"LLM returned non-JSON: {exc}; raw={llm_response!r}")
    parsed = {}

# Network I/O where transient failure is a normal operating condition
try:
    resp = await embedding_client.embeddings.create(input=[text], model=model)
except (TimeoutError, httpx.ConnectError) as exc:
    raise EmbeddingUnavailableError(f"Embedding service unreachable: {exc}") from exc
```

When reviewing: for each `try-except`, ask whether the caught exception is caused by something outside the current codebase. If the only way it fires is a bug in internal code, remove the block and let it crash.

---

## Rule 7 — Avoid nested dictionary retrieval chains

Nested retrieval chains make code hard to inspect and debug because several lookups are compressed into one expression. One statement should perform at most one `.get(...)` lookup. Split chained retrieval into named intermediate variables so a debugger or log statement can inspect each layer directly.

This rule does not make `.get(...)` acceptable for internal data that should always exist. Rule 5 still applies: use plain indexing for required internal fields so missing data crashes at the real bug site.

**Wrong:**
```python
user_memory_context = (
    (global_state.get("rag_result", {}).get("user_image") or {})
    .get("user_memory_context")
    or {}
)
```

**Right:**
```python
rag_result = global_state["rag_result"]
user_image = rag_result["user_image"]
user_memory_context = user_image["user_memory_context"]
```

When reviewing: flag any statement that chains multiple `.get(...)` calls. Ask for intermediate variables with names that explain the shape being traversed.

---

## Rule 8 — Return values or attributes, not call expressions

Returning a function call directly makes debugging harder because the produced value has no local name and cannot be inspected before leaving the function. Store the result in a clearly named local variable, then return that value or return a simple attribute/index access.

**Wrong:**
```python
return project_user_memory_units(units, budget=budget)
```

**Right:**
```python
projected_units = project_user_memory_units(units, budget=budget)
return projected_units
```

**Acceptable:**
```python
return profile.affinity
return profile["affinity"]
```

When reviewing: flag `return some_function(...)`, `return await some_function(...)`, and other compound return expressions. Prefer assigning the expression to a named local before returning it.

---

## Rule 9 — Use `_` for unused tuple values

When unpacking a tuple, use `_` for any value that is intentionally unused. A named variable signals that the value matters later; using `_` keeps the reader focused on the values that are actually part of the function's behavior.

**Wrong:**
```python
hydrated_profile, _memory_blocks = await user_image_retriever_agent(
    global_user_id,
    user_profile=context.get("user_profile"),
    input_embedding=input_embedding,
    include_semantic=True,
)
```

**Right:**
```python
hydrated_profile, _ = await user_image_retriever_agent(
    global_user_id,
    user_profile=context.get("user_profile"),
    input_embedding=input_embedding,
    include_semantic=True,
)
```

When reviewing: after tuple unpacking, verify each named value is used. Replace intentionally unused values with `_`.

---

## Rule 10 — Exception handlers must include the actual exception message

Every `except` block that logs, prints, wraps, or otherwise reports a failure must include the actual exception message. Bind the exception with `as exc` and include `{exc}` in the emitted message. For broad boundary catches that use `logger.exception(...)`, still include `{exc}` in the message; the traceback alone is not enough for this project.

**Wrong:**
```python
try:
    await call_external_service()
except TimeoutError:
    logger.exception("External service failed")
```

**Right:**
```python
try:
    await call_external_service()
except TimeoutError as exc:
    logger.exception(f"External service failed: {exc}")
```

**Also right when wrapping:**
```python
try:
    await save_memory(doc)
except PyMongoError as exc:
    raise RuntimeError(f"Failed to save memory: {exc}") from exc
```

When reviewing: every `except` body must make the concrete exception text visible in logs, prints, or raised wrapper messages unless the exception is intentionally re-raised immediately with a bare `raise`.

---

## Rule 11 — Use f-strings instead of `%s`-style substitution

Use f-strings for string interpolation. Do not use legacy `%` formatting, and do not rely on logging's comma-argument `%s` substitution style. This project prioritizes immediate readability over deferred logging interpolation.

**Wrong:**
```python
logger.info("User %s affinity changed to %d", user_id, affinity)
message = "User %s" % user_id
```

**Right:**
```python
logger.info(f"User {user_id} affinity changed to {affinity}")
message = f"User {user_id}"
```

When reviewing: flag `%` string formatting and logging calls that pass a format string plus interpolation arguments. Rewrite them as f-strings.

---

## Rule 12 — Keep LLM prompt, instance, and handler adjacent

For LLM-backed stages, keep the three stage-defining parts next to each other in this order:

```text
prompt constant
LLM instance
LLM handler function
```

This makes it possible to review a prompt together with the exact model configuration, payload construction, parser, and structural validation that consume it.

Do not group unrelated prompt constants at the top of a file while placing the handlers far below. Do not route distinct LLM stages through a generic helper that hides the prompt/model/payload/parser relationship.

Prompt constants for structured LLM stages must include explicit generation guidance plus `# Input Format` and `# Output Format` sections. Use a section such as `# Generation Procedure` or `# Thinking Steps` to tell the local LLM how to inspect inputs and choose output fields. The format sections must match the handler's actual JSON payload and the validator's expected parsed output. Treat missing or stale guidance/format sections as a style violation because they hide prompt/handler contract drift.

Prompt constants must not hard-code a character's concrete name unless that exact name is part of a test fixture or fixed source text. For reusable runtime prompts, decide whether the LLM actually needs the character name:

- If the prompt only needs role ownership, use role-neutral wording such as "the active character", "the character", or "the speaker". This is preferred for subjective appraisal fields, internal reasoning fields, memory schemas, RAG summaries, and dialog-adjacent instructions where a name can shift voice.
- If identity distinction is required, use the runtime `.format(...)` variable such as `{character_name}` and pass `state["character_profile"]["name"]` in the handler, following existing cognition prompt style.
- Do not put learned names from logs or examples into schema examples. Examples are sticky for local LLMs and can accidentally anchor persona voice or subject/object interpretation.

**Wrong:**
```python
_FIRST_PROMPT = "..."
_SECOND_PROMPT = "..."
_shared_llm = get_llm(...)

async def first_handler(state):
    system_prompt = SystemMessage(content=_FIRST_PROMPT)
    human_message = HumanMessage(content=json.dumps(state, ensure_ascii=False))
    response = await _shared_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)
```

**Right:**
```python
_FIRST_PROMPT = "..."
_first_llm = get_llm(...)
async def first_handler(state):
    system_prompt = SystemMessage(content=_FIRST_PROMPT)
    human_message = HumanMessage(content=json.dumps(state, ensure_ascii=False))
    response = await _first_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)
    ...
```

When reviewing: for every LLM-backed handler, verify the prompt, LLM object, and handler are adjacent, verify structured JSON output uses the repo's established parser such as `parse_llm_json_output`, and scan prompt examples/instructions for hard-coded character names that should be role-neutral or runtime-formatted.

---

## Rule 13 — Search for existing functions before adding a new one

Before adding any new function or method, search the project for similar behavior first, including private helpers prefixed with `_`. Reusing an existing function keeps behavior consistent, avoids parallel implementations drifting apart, and respects the local abstractions that already exist in the codebase.

Use `rg` or another fast project-wide search to look for nearby concepts, verbs, return shapes, and domain nouns. Search outside the current file when the behavior is not obviously local. Prefer adapting the call site to an existing helper when the existing function already expresses the same concept, even if its name or privacy level means you need to read its callers before deciding.

**Wrong:**
```python
def normalize_memory_text(text: str) -> str:
    return " ".join(text.strip().split())
```

when the project already has:

```python
def _normalize_text_for_memory(text: str) -> str:
    ...
```

**Right:**
```python
normalized_text = _normalize_text_for_memory(raw_text)
```

When reviewing: for each newly added function or method, ask what project search was done and whether an existing public or private function already covers the behavior. Flag duplicate implementations unless there is a clear reason the existing helper cannot be reused.

---

## Rule 14 — Add helper functions only when the abstraction earns its place

Helper functions are not automatically cleaner. A small block of straightforward code is often easier to read inline than behind a new name, especially when it is used once or twice. Add a helper only when it removes real repetition or clarifies a genuinely complex operation.

As a default threshold, require the same logic block to be used at least three times before extracting a helper. A helper used fewer than three times needs a concrete justification such as isolating a complex domain concept, giving a testable name to non-obvious behavior, or matching an established local pattern. Without that justification, keep the code direct and readable at the call site.

**Wrong:**
```python
def _build_result(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}

result = _build_result("ok", "Saved")
```

**Right:**
```python
result = {"status": "ok", "message": "Saved"}
```

**Also right when repetition justifies it:**
```python
def _build_result(status: str, message: str) -> dict[str, str]:
    """Build the status payload shared by save, delete, and refresh handlers."""
    return {"status": status, "message": message}
```

When reviewing: challenge every newly added helper. If it is not used at least three times, require a clear readability, testability, or existing-pattern justification. Otherwise inline the logic and keep the code simple.

---

## Review workflow

When asked to review a file or selection:

1. Read the code in full.
2. First check for PEP 8 violations, then check the fourteen project-specific rules.
3. For each violation, list the line reference, the violated standard, and a one-sentence explanation of why it violates the rule.
4. Propose the corrected version inline.
5. If no violations are found, say so explicitly.

When writing new code:

Apply PEP 8 and all fourteen project-specific rules before producing any output. If a rule conflict arises or a genuine exception is needed, surface it to the user with the reasoning rather than silently picking one path.

See `references/examples.md` for additional annotated before/after examples.
