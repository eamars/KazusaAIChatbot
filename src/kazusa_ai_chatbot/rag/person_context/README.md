# Person Context ICD

`kazusa_ai_chatbot.rag.person_context` owns user, character, profile, image,
and relationship-like factual context retrieval.

## Public Contract

```python
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent

await PersonContextAgent().run(
    task: str,
    context: dict,
    max_attempts: int = 1,
)
```

The returned shape remains the standard capability result:

```python
{
    "resolved": bool,
    "result": {
        "selected_summary": str,
        "capability": "person_context",
        "primary_worker": str,
        "supporting_workers": list[str],
        "source_policy": str,
        "resolved_refs": list[dict],
        "projection_payload": dict,
        "worker_payloads": dict,
        "evidence": list[str],
        "missing_context": list[str],
        "conflicts": list[str],
        "observation_candidates": list[dict],
        "source_hints": list[dict],
    },
    "attempts": int,
    "cache": dict,
}
```

## Semantic Ownership

Person context resolves named people, enumerates users by display-name or
participant metadata predicates, reads user and character profile bundles,
hydrates approved user image/profile context, and produces factual
relationship/profile rankings.

## Non-Ownership

It does not search message content, durable memory, live public facts, recall
state, or decide persona stance. When a task needs message evidence, memory
evidence, or external facts, it returns missing context or routes internally
only through the existing approved worker chain.

## Internal Flow

```text
selector -> selected person worker -> projection -> capability result
```

Deterministic target resolution handles current user, active character, known
person refs, and display-name to profile chains where supported. The selector
LLM chooses a person-context mode only when deterministic planning does not
resolve the mode.

## Module Ownership

| Module | Ownership |
|---|---|
| `agent.py` | Capability wrapper and run orchestration. |
| `selector.py` | Deterministic and LLM mode selection for lookup, profile, user-list, and relationship paths. |
| `contracts.py` | Person-context result helpers and uncached capability contract. |
| `projection.py` | Person refs, profile target context, profile summaries, people summaries, and profile-kind projection. |
| `workers/lookup.py` | One-person display-name identity lookup. |
| `workers/list.py` | User enumeration by display-name or participant metadata. |
| `workers/profile.py` | User or character profile bundle retrieval. |
| `workers/relationship.py` | Factual relationship/profile rankings. |
| `workers/image.py` | User image and memory-context hydration for profile reads. |

## Worker Roster

- `user_lookup_agent`
- `user_list_agent`
- `user_profile_agent`
- `relationship_agent`
- `user_image_retriever_agent`

Worker names, cache names, result payloads, and fact-source metadata remain
unchanged.

## Cache Policy

The top-level capability is not cached and reports
`capability_orchestrator_uncached`. Worker cache policy remains worker-local and
uses Cache 2 dependency invalidation for profile and relationship sources.

## LLM And Prompt Policy

Prompt constants, LLM instances, message construction, JSON parsing, retry
behavior, and call ordering are preserved from the flat modules. Stable
instructions stay in `SystemMessage`; runtime task/context stays in
`HumanMessage`.

## Verification

- `py_compile` every changed Python file in this package.
- Static grep for deleted flat person-context module import paths returns no
  production-source matches.
- Prompt stability audit confirms selector and worker prompt payloads are
  unchanged.
- Person-context deterministic tests remain the behavioral authority.
