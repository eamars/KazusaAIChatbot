---
name: test-style-and-execution
description: Write, refactor, and run tests in this repo using the project testing contract. Use this skill whenever adding or changing pytest tests, refactoring test style, deciding between real LLM tests and patched unit tests, testing graph/subgraph behavior, or running tests. It covers both how tests should be written and how regular versus real LLM tests must be executed and inspected.
---

# Test Style and Execution

Use this skill when writing, refactoring, reviewing, or running tests for this project.

The goal is to keep tests simple, inspectable, and matched to the kind of behavior being tested. Real LLM tests are evidence-producing contract checks. Patched tests are deterministic checks for graph plumbing and handoff behavior.

## Core Testing Contract

Separate test strategy by responsibility:

- Use **real LLM tests** for prompt behavior and model-facing graph behavior.
- Use **patched LLM tests** for handoff between graphs or subgraphs.
- Use **pure deterministic tests** for validators, parsers, state reducers, schema mapping, formatting, permissions, and limit logic.

Do not use mocked LLM responses to claim that a prompt works. Do not use live model variance to test deterministic orchestration code.

## Test Taxonomy

### Real LLM Node Tests

Use these when testing an individual LLM call inside a graph or subgraph.

- Test each LLM node individually.
- Use 2-5 realistic inputs per LLM node.
- Include normal cases and at least one boundary, ambiguous, or unsupported case when relevant.
- Assert basic structure and contract requirements, not exact phrasing.
- Emit enough logs for a human or AI agent to inspect the result without rerunning.

These tests answer: "Can this prompt/model contract perform its job with realistic input?"

### Real LLM Graph Tests

Use these when testing a complete graph or subgraph with live model behavior.

- Add at least 1 real LLM test for each graph or subgraph that has model-facing behavior.
- Prefer a representative high-value path over broad exhaustive coverage.
- Keep assertions loose and structural.
- Log the full path clearly enough to diagnose whether failure came from a node, state transition, routing decision, retrieval step, or final output.

These tests answer: "Do the real LLM nodes cooperate well enough in the full graph?"

### Patched LLM Handoff Tests

Use these when testing handoff between graphs, subgraphs, or orchestration stages.

- Patch real LLM responses with small explicit fixtures.
- Verify the selected next graph/subgraph, payload shape, state merge, error flow, and downstream calls.
- Keep these tests deterministic, fast, and suitable for normal batch execution.
- Include failure and edge paths here rather than relying on live model tests to stumble into them.

These tests answer: "Does the code move state and control flow across boundaries correctly?"

### Deterministic Unit Tests

Use these for code that does not need an LLM to be meaningful.

- Test validators, parsers, mappers, reducers, permission checks, and formatting directly.
- Prefer direct inputs and outputs.
- Keep pass criteria strict when behavior is deterministic.

## Real LLM Pass Criteria

Real LLM tests should be loose by default. They are diagnostic and qualitative, not brittle golden-output checks.

The pass criteria includes the agent's judgment, not only pytest status. After inspecting the logs, decide whether the behavior is acceptable for the contract under test. A test can have passing assertions while still needing attention if the model output is confused, brittle, off-contract, or only accidentally correct.

Good basic assertions:

- No exception is raised.
- The response is non-empty.
- Required top-level fields are present.
- Structured output parses when the contract requires structured output.
- The selected route, status, or decision is one of the allowed values.
- The model does not refuse an answerable case.
- The model does refuse or mark unsupported cases when that is the expected contract.

Avoid by default:

- Exact prose matches.
- Full JSON equality unless testing a deterministic wrapper.
- Over-checking every inferred field.
- Assertions that make harmless wording or ordering changes fail the test.

If the user asks for stricter quality gates, add them deliberately and explain what risk they guard.

## Required Logs For Real LLM Tests

Every real LLM test must emit enough information for the user or an AI agent to inspect the behavior after the run.

Prefer writing the real LLM trace to a file for post-run analysis. Console output is useful while watching a test, but do not rely on the terminal output buffer as the only record. A durable log file should contain the case input, raw model output, parsed output, important graph state, and judgment notes when practical.

Include, when available:

- Test name and case id.
- Model/provider/config.
- Input user message or graph state.
- Rendered prompt, prompt id, or prompt version.
- Raw LLM output.
- Parsed output.
- Selected graph, subgraph, route, or tool.
- Important intermediate graph state.
- Final graph state or final answer.
- Basic pass/fail reason.

Prefer clear `logger.info(...)` output or an existing project logging pattern, with a file handler or test artifact path for real LLM traces. Keep logs readable for one case at a time.

## Execution Rules

Regular deterministic tests may be run in batches.

Allowed:

```powershell
pytest
pytest tests\some_area -q
pytest tests\some_file.py -q
```

Real LLM tests must be run one by one and inspected one by one.

Required workflow:

```text
run one real LLM test
inspect the emitted logs/output
judge whether the behavior is acceptable
then run the next real LLM test
```

Preferred command shape:

```powershell
pytest path\to\test_file.py::test_specific_real_llm_case -q -s
```

Do not batch-run real LLM tests for a simple green/red summary. If a real LLM test fails, hangs, or produces suspicious output, stop and interpret that case before continuing.

If regular test runs reveal unrelated failures, do not ignore them by default. Identify them as unrelated, summarize the failure briefly, and ask the user whether to address them now or continue with the current task. Do not silently expand the work unless the failure blocks the requested change.

## Style Rules

- Write plain pytest tests with descriptive names.
- Prefer one test function per case.
- Prefer copy-paste structure over clever abstractions when inspection matters.
- Avoid parameterized tests for real LLM cases unless the user explicitly asks for them.
- Use one small helper only when it removes obvious repetition without hiding the case details.
- Keep assertions minimal and structural in manual-inspection tests.
- Keep deterministic tests stricter and more compact.

If the user is replacing the test style, do not infer the desired baseline from old tests unless they explicitly ask you to inspect them.

## Decision Guide

Use real LLM tests when the task is about:

- prompt quality
- structured LLM output quality
- routing quality
- graph or subgraph input-output behavior
- refusal or unsupported-intent behavior
- retrieval quality
- supervisor or agent behavior
- real conversation history behavior

Use patched LLM tests when the task is about:

- graph-to-graph handoff
- subgraph orchestration
- state passing
- state merging
- error paths
- fallback behavior
- downstream calls

Use deterministic tests when the task is about:

- pure data transformation
- schema validation
- deterministic branching
- permission, scope, or limit enforcement
- formatting-only logic

## Example Pattern

Preferred real LLM case:

```python
async def test_planner_live_routes_user_list_request(live_env: dict) -> None:
    result = await call_planner(
        message="Show me users whose display name ends with 子.",
        env=live_env,
    )

    assert isinstance(result, dict)
    assert result.get("route") == "user_list"

    logger.info("input=%s", "Show me users whose display name ends with 子.")
    logger.info("raw_output=%s", result.get("raw_output"))
    logger.info("parsed=%s", result)
```

Preferred patched handoff case:

```python
async def test_dispatcher_hands_user_list_plan_to_user_list_subgraph() -> None:
    planner_result = {
        "route": "user_list",
        "constraints": [{"field": "display_name", "op": "ends_with", "value": "子"}],
    }

    result = await dispatch_from_planner_result(planner_result)

    assert result.next_graph == "user_list"
    assert result.payload["constraints"] == planner_result["constraints"]
```

Avoid when inspection matters:

```python
@pytest.mark.parametrize(...)
async def test_many_live_cases(...):
    ...
```

## Communication

When reporting test work:

- Say whether tests were regular, patched LLM, or real LLM.
- For real LLM tests, say that each was run individually and inspected.
- Summarize the observed model behavior and give your judgment on whether it satisfies the intended contract, not only pass/fail status.
- Mention if logs were insufficient and improve them before relying on the result.
- If unrelated failures appeared, say what failed and whether the user chose to address or defer them.
