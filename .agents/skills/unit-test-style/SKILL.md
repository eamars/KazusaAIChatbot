---
name: unit-test-style
description: Write and edit tests in this repo using a simple, inspectable style. Use this skill whenever adding or changing pytest tests, especially when the user wants one test per case, wants to inspect outputs one by one, or wants real LLM / real database tests instead of mocks. Default to real LLM tests when the environment can support them, and prefer basic copy-paste test structure over clever abstractions.
---

# Unit Test Style

Use this skill when writing or modifying tests for this project.

The goal is simple tests that are easy to inspect, easy to run one by one, and easy to debug.

## Core defaults

- Default to real LLM tests when possible.
- If real database access is available and the user wants realistic behavior, use the real database too.
- Prefer one test function per case.
- Prefer copy and paste over fancy abstractions.
- Keep helpers small and boring.
- Avoid parameterized tests unless the user explicitly wants them.

## Style rules

- Write plain pytest tests with descriptive names.
- If there are many cases, duplicate the test body and change only the input values if that makes inspection easier.
- It is OK to use one small shared helper when it removes obvious repetition, but do not build a mini framework.
- Log the important output for manual inspection when the test is qualitative.
- Do not add strict pass criteria for subjective LLM quality unless the user asks for them.

## Real LLM preference

When choosing between mocked tests and real LLM tests:

1. If the user cares about output quality, routing quality, retrieval quality, or prompt behavior, prefer a real LLM test.
2. If the feature depends on real database contents or real retrieval, prefer a real database test too.
3. Only fall back to mocks when real execution is impossible, too slow for the task, or the user explicitly asks for unit-only isolation.

## Test organization

- Put one scenario in one test function.
- Give each test a name that clearly says what it is checking.
- If the outputs need human review, make the test emit enough logs to inspect:
  - input
  - context
  - answer
  - important intermediate fields
- Keep the assertions minimal and structural in manual-inspection tests.

## Good patterns

Preferred:

```python
async def test_call_rag_supervisor_live_opinion_small_pliers(live_env: dict) -> None:
    result = await call_rag_supervisor(...)

    assert isinstance(result, dict)
    assert "answer" in result

    logger.info("answer=%s known_facts=%s", result["answer"], result["known_facts"])
```

Avoid when inspection matters:

```python
@pytest.mark.parametrize(...)
async def test_many_cases(...):
    ...
```

## Running tests

- When the user wants inspection, run tests one by one.
- Prefer commands like:
  - `pytest path\\to\\test_file.py::test_name -q`
- If there are many live tests, run them sequentially so logs stay readable.

## Decision guide

Use real LLM tests first when the task is about:

- prompt quality
- retrieval quality
- routing quality
- supervisor behavior
- multi-step agent behavior
- real conversation history behavior

Use mocked tests first when the task is about:

- pure data transformation
- deterministic branching
- error handling on internal helpers
- formatting-only logic

## Communication style

- Be direct.
- Keep the test basic.
- Don’t over-engineer.
- If a test is mainly for manual review, say so clearly.
