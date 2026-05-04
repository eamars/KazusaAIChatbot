# Positive Python Constraints

Positive constraints describe the preferred way to write Python in this
project. Apply every `P-XXX` constraint together with PEP 8 and the negative
constraints.

---

## P-001 -- Keep imports at the top of the module

Imports make module dependencies visible. Place all `import` and `from ...
import` statements after the module docstring, before runtime code.

Test code, pytest fixtures/functions, and `if __name__ == "__main__"` blocks may
use inline imports when they need heavy or optional dependencies without
affecting public module load cost.

**Wrong:**
```python
async def save_memory(doc):
    from datetime import datetime
    ...
```

**Right:**
```python
from datetime import datetime


async def save_memory(doc):
    ...
```

When reviewing, scan function and method bodies for hidden imports outside test
or harness code.

---

## P-002 -- Keep `try` blocks limited to the risky statement

A `try` block should answer "what specifically might fail here?" Keep it scoped
to the statement or statements that directly raise the caught exception. Move
logging, unrelated assignments, and follow-up processing outside the block.

**Wrong:**
```python
try:
    result = parse_llm_output(response)
    logger.info(f"Parsed {len(result)} facts")
    enriched = enrich(result)
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

When reviewing, verify every statement inside a `try` is directly related to
the exception being caught.

---

## P-003 -- Give every non-trivial function a useful docstring

Type annotations explain shape; docstrings explain purpose. Every non-trivial
function should document why it exists, what each argument means conceptually,
and what the return value represents.

One-liner helpers whose name already makes the contract self-evident may skip
the docstring.

**Wrong:**
```python
async def update_affinity(global_user_id: str, delta: int) -> int:
    """Update affinity."""
    ...
```

**Right:**
```python
async def update_affinity(global_user_id: str, delta: int) -> int:
    """Apply a signed delta to the user's affinity score, clamped to 0-1000.

    Args:
        global_user_id: Internal UUID identifying the user in user_profiles.
        delta: Signed integer; positive values increase affinity, negative
            values decrease it.

    Returns:
        The new affinity value after clamping.
    """
    ...
```

When reviewing, flag missing docstrings and docstrings that omit purpose, args,
or return value for non-trivial functions.

---

## P-004 -- Put defaults at the point of definition

Defaults should live in one authoritative place: a config constant, DB write,
schema, TypedDict, or model definition. Read required internal data with plain
indexing so missing data crashes at the bug site instead of being hidden by a
fallback.

**Wrong:**
```python
affinity = doc.get("affinity", 500)
affinity = state.get("affinity", 500)
affinity = profile.get("affinity", 500)
```

**Right:**
```python
AFFINITY_DEFAULT = 500

new_profile = {"affinity": AFFINITY_DEFAULT, ...}
await db.user_profiles.insert_one(new_profile)

affinity = profile["affinity"]
```

`.get(key, fallback)` is appropriate for genuinely external or optional data:
LLM JSON output, API responses, user-provided dictionaries, or optional config.

---

## P-005 -- Name computed return values before returning them

Returning a function call directly makes debugging harder because the produced
value cannot be inspected before leaving the function. Store computed results in
a clearly named local variable, then return that variable.

**Wrong:**
```python
return project_user_memory_units(units, budget=budget)
```

**Right:**
```python
projected_units = project_user_memory_units(units, budget=budget)
return projected_units
```

Simple attribute or index returns are fine:

```python
return profile.affinity
return profile["affinity"]
```

---

## P-006 -- Use `_` for intentionally unused tuple values

Named variables signal that their values matter later. When unpacking a tuple,
use `_` for any value that is intentionally unused.

**Wrong:**
```python
hydrated_profile, _memory_blocks = await user_image_retriever_agent(
    global_user_id,
    user_profile=user_profile,
)
```

**Right:**
```python
hydrated_profile, _ = await user_image_retriever_agent(
    global_user_id,
    user_profile=user_profile,
)
```

When reviewing, check tuple unpacking and replace intentionally unused named
values with `_`.

---

## P-007 -- Include the exception message when reporting failures

Every handler that logs, prints, wraps, or otherwise reports a failure should
include the concrete exception text. Bind the exception with `as exc` and
include `{exc}` in the emitted message.

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

Immediate bare re-raise is the exception:

```python
except TimeoutError:
    raise
```

---

## P-008 -- Use f-strings for interpolation

Use f-strings for string interpolation. They keep the rendered message close to
the values being inserted and match the project's logging style.

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

---

## P-009 -- Keep LLM prompt, instance, and handler adjacent

For LLM-backed stages, keep the prompt constant, LLM instance, and handler
function next to each other in this order:

```text
prompt constant
LLM instance
LLM handler function
```

This lets reviewers inspect the prompt together with the model configuration,
payload construction, parser, and structural validation that consume it.

Structured LLM prompts must include explicit generation guidance plus
`# Input Format` and `# Output Format` sections. Use a section such as
`# Generation Procedure` or `# Thinking Steps` to tell the local LLM how to
inspect inputs and choose output fields. The format sections must match the
handler's actual JSON payload and validator expectations.

**Wrong:**
```python
_FIRST_PROMPT = "..."
_SECOND_PROMPT = "..."
_shared_llm = get_llm(...)


async def first_handler(state):
    response = await _shared_llm.ainvoke(...)
```

**Right:**
```python
_FIRST_PROMPT = "..."
_first_llm = get_llm(...)


async def first_handler(state):
    response = await _first_llm.ainvoke(...)
```

When reviewing, verify each LLM-backed handler keeps its prompt, model, payload,
parser, and validation contract together.

---

## P-010 -- Search for existing functions before adding a new one

Before adding any new function or method, search the project for similar
behavior, including private helpers prefixed with `_`. Reusing existing
behavior keeps contracts consistent and avoids parallel implementations
drifting apart.

Use `rg` or another fast project-wide search for nearby concepts, verbs, return
shapes, and domain nouns. Search outside the current file when the behavior is
not obviously local.

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

When reviewing, ask what project search was done and whether an existing public
or private helper already covers the behavior.

---

## P-011 -- Add helpers and abstractions only when they earn their place

Simple code that solves today's requirement is preferred over architecture for
tomorrow's possible requirement. Add a helper, class, strategy object, protocol,
config layer, cache, validator, or optional behavior only when it removes real
repetition, clarifies a current domain concept, matches an established local
pattern, or is required by the task in front of you.

As a default threshold, require the same non-trivial logic block to appear at
least three times before extracting a helper. A helper used fewer than three
times needs a concrete justification, such as isolating a complex domain
concept, giving a testable name to non-obvious behavior, or matching an
established local pattern.

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

**Also wrong:**
```python
class DiscountStrategy(Protocol):
    def calculate(self, amount: float) -> float:
        ...


class PercentageDiscount:
    def __init__(self, percent: float) -> None:
        self.percent = percent

    def calculate(self, amount: float) -> float:
        return amount * (self.percent / 100)
```

when the current requirement is only:

```python
def calculate_discount(amount: float, percent: float) -> float:
    """Calculate the discount amount for a percentage discount."""

    discount = amount * (percent / 100)
    return discount
```

**Also right when repetition justifies it:**
```python
def _build_result(status: str, message: str) -> dict[str, str]:
    """Build the status payload shared by save, delete, and refresh handlers."""
    return {"status": status, "message": message}
```

When reviewing, challenge every new helper or abstraction and require it to
earn the extra names, files, branches, and tests readers must follow. If a
future requirement would justify more structure, refactor when that requirement
arrives instead of pre-building the structure now.

---

## P-012 -- Describe code with durable runtime or domain language

Comments and docstrings should explain stable behavior, invariants, domain
meaning, or implementation tradeoffs that will still make sense after the
current edit plan is gone. Use names and explanations tied to what the code does
at runtime, not to how the work was split during one implementation batch.

**Wrong:**
```python
def _keyword_text_filter(pattern: str) -> dict[str, Any]:
    """Build the Phase-F regex filter for typed conversation text."""

    regex_filter = {"$regex": pattern, "$options": "i"}
    filter_doc: dict[str, Any] = {"body_text": regex_filter}
    return filter_doc
```

**Right:**
```python
def _keyword_text_filter(pattern: str) -> dict[str, Any]:
    """Build a case-insensitive filter for typed conversation text."""

    regex_filter = {"$regex": pattern, "$options": "i"}
    filter_doc: dict[str, Any] = {"body_text": regex_filter}
    return filter_doc
```

When reviewing comments and docstrings, ask whether the wording will still be
clear six months later to someone who has never read the implementation plan.

---

## P-013 -- Surface risky assumptions before changing behavior

Before changing Python behavior, identify any assumption that affects scope,
privacy, data shape, persistence location, API contract, or which performance
metric matters. Use existing code, tests, schemas, and call sites to resolve
ordinary questions yourself. If a decision is still ambiguous and the wrong
choice would create user-visible behavior, data exposure, or avoidable rework,
ask the user or state the assumption explicitly before editing.

**Wrong:**
```python
def export_users() -> str:
    """Export all users to a local JSON file."""

    users = User.query.all()
    with open("users.json", "w") as file_handle:
        json.dump([user.to_dict() for user in users], file_handle)
    message = f"Exported {len(users)} users"
    return message
```

This silently chooses all users, all fields, a local file path, and synchronous
execution even though each choice changes product behavior and risk.

**Right before editing:**
```text
Assumptions to resolve: whether export means API response or file download,
which user subset and fields are allowed, whether sensitive fields must be
excluded, and whether expected volume needs pagination or a background job.
```

When reviewing, flag changes that silently pick a scope, destination, data
field set, or optimization target that the request did not specify and the code
did not already establish.

---

## P-014 -- Keep requested changes surgical and style-compatible

A code change should fit the request's blast radius. Change the specific lines
needed to satisfy the request, preserve the surrounding style, and avoid
opportunistic improvements that are not required for the current bug or feature.

Match local conventions for quote style, typing density, comments, control
flow, naming, and return shape unless the requested change directly requires a
different pattern.

**Wrong:**
```python
def validate_user(user_data: dict[str, str]) -> bool:
    """Validate user data."""

    email = user_data.get("email", "").strip()
    if not email:
        raise ValueError("Email required")
    if "@" not in email or "." not in email.split("@")[1]:
        raise ValueError("Invalid email")

    username = user_data.get("username", "").strip()
    if len(username) < 3:
        raise ValueError("Username too short")
    return True
```

when the task was only to stop empty email strings from crashing an existing
validator.

**Right:**
```python
email = user_data.get("email", "")
if not email or not email.strip():
    raise ValueError("Email required")

if "@" not in email:
    raise ValueError("Invalid email")
```

When reviewing, separate necessary changes from drive-by refactors. A style
cleanup, new type annotation, docstring, stronger validation rule, or rewritten
control flow must be justified by the task, not by general preference.

---

## P-015 -- Tie implementation to verifiable outcomes

For bug fixes, reproduce the failure before changing the implementation when a
reasonable test or local command exists. For new behavior, define the observable
success criteria before implementation and verify the smallest useful slice
before broadening the change.

Prefer incremental checks: targeted test first, then adjacent tests for shared
behavior, then broader suites when the touched code has wider reach. If the
environment prevents running a check, report the blocker and the verification
that remains undone.

**Wrong:**
```text
Fix authentication by reviewing the code, making improvements, and testing.
```

**Right:**
```text
Reproduce: password change leaves the old session valid.
Implement: invalidate sessions when the password version changes.
Verify: targeted session test fails before the fix and passes after it; existing
auth tests still pass.
```

When reviewing, ask whether the change has a concrete before/after condition
and whether the submitted verification proves that condition rather than merely
showing that code was edited.
