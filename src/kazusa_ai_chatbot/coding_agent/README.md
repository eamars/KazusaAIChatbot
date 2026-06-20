# Coding Agent ICD

The `coding_agent` package contains standalone code-task modules that can be
called directly by tests and future background workers.

Current implemented surface:

```python
from kazusa_ai_chatbot.coding_agent.code_fetching import run
```

`code_fetching.run(...)` resolves a supported code source into a local source
contract. It does not read files to answer questions, write patches, execute
project commands, or integrate with Kazusa service/background-worker runtime.

Implemented subagents:

- `code_fetching`: resolves public GitHub and explicit local-checkout sources.

Deferred subagents:

- `code_reading`
- `code_writing`
- `code_executing`

Managed checkouts live under the caller-supplied coding workspace root. Direct
standalone use may fall back to an OS temp workspace, but future worker
integration must pass an explicit configured workspace.
