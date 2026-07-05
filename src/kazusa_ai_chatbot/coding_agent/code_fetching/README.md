# Code Fetching ICD

`code_fetching` resolves one supported source-code location into a safe local
repository contract.

Public entrypoint:

```python
from kazusa_ai_chatbot.coding_agent.code_fetching import run

result = await run({
    "question": "Explain https://github.com/owner/repo",
    "workspace_root": "C:/path/to/coding-workspace",
})
```

Supported source inputs:

- `source_url` or `repo_url` with public GitHub repository, `.git`, tree, blob,
  or raw file URLs. Explicit source fields are trusted request fields and are
  resolved without a source-intake LLM call.
- `repo_hint` as `owner/repo`.
- public GitHub URLs embedded in `question`, including markdown links. These
  go through a source-intake specialist on `CODING_AGENT_PM_LLM`, then a
  deterministic resolver validates anchoring, provider grammar, cardinality,
  scope, and public status mapping.
- `local_root_hint` pointing at an existing GitHub-backed local checkout.
- `local_path_hint` pointing at a file or directory inside an existing
  GitHub-backed local checkout.

Unsupported inputs return `rejected` or `needs_user_input`; they are not guessed
or fetched through generic web browsing. Unsupported inputs include SSH URLs,
credential-bearing URLs, non-GitHub providers, GitHub issues, pull requests,
discussions, archives, Gists, package registry names, paste URLs, and arbitrary
raw HTTP files.

Question-text source resolution is LLM-first for semantic extraction and
deterministic for validation. The source-intake specialist extracts visible
source mentions, roles, and source families. The resolver then accepts only
anchored mentions, rejects unsupported providers without probing them, preserves
percent-encoded URLs, asks for clarification on ambiguous primary sources, and
keeps invalid explicit source fields authoritative instead of replacing them
with a fallback URL from the question. GitHub issue, pull-request, and
discussion URLs are supported only as repository identity clues when the intake
role marks them as reference-only for repository-level analysis; target-content
thread analysis remains unsupported.

GitHub repository, tree, and blob scopes resolve through a managed clone.
GitHub raw-file URLs resolve through a managed single-file download under the
configured workspace. Raw download metadata uses `storage_kind:
"managed_download"` and a `raw-sha256:<hash>` content identity because the raw
HTTP response does not expose a Git commit. Existing matching managed downloads
are reused and not auto-refreshed.

Managed clone and raw-download storage use compact hash-named paths under the
configured workspace. Repository identity stays in metadata instead of nested
owner/repo/ref directories. Supported-provider access failures, such as clone
failure, invalid requested ref, missing scoped path, inaccessible raw URL, or
raw size limit, return corrective `needs_user_input` outcomes. Local managed
workspace/preparation failures return generic public-safe `failed` outcomes.
Local roots, workspace roots, cache keys, raw command stderr, and cleanup paths
stay out of public results.

GitHub `tree`, `blob`, and raw-file scopes must exist in the resolved local
root. Missing scoped paths return `needs_user_input` instead of handing an
invalid path to reading. Local checkout scopes use public-safe
`local://github/<owner>/<repo>` source labels; `repository.local_root` remains
the internal filesystem root used by downstream readers.

Result shape:

```python
{
    "status": "succeeded | failed | needs_user_input | rejected",
    "message": str,
    "repository": CodeRepositoryRef | None,
    "source_scope": CodeSourceScope | None,
    "limitations": list[str],
    "trace_summary": list[str],
}
```

The subagent does not answer code questions, read source files for evidence,
write patches, run project commands, install packages, or call web agents.
