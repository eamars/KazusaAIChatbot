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
- pasted inline code blocks, inline diffs, and anchored inline code fragments
  embedded in `question`. Source intake identifies semantic roles and short
  visible anchors; deterministic code extracts exact text from the original
  task and materializes it as a managed read-only inline bundle.
- trusted `inline_sources` supplied directly by a caller that already owns the
  source text.
- `local_root_hint` pointing at an existing GitHub-backed local checkout.
- `local_path_hint` pointing at a file or directory inside an existing
  GitHub-backed local checkout.

Unsupported inputs return `rejected` or `needs_user_input`; they are not guessed
or fetched through generic web browsing. Unsupported inputs include SSH URLs,
credential-bearing URLs, non-GitHub providers, GitHub issues, pull requests,
discussions, archives, Gists, package registry names, paste URLs, arbitrary
raw HTTP files, image-only code screenshots, secret-like pasted code, oversized
inline snippets, and too many independent inline fragments.

Question-text source resolution is LLM-first for semantic extraction and
deterministic for validation. The source-intake specialist extracts visible
source mentions, roles, source families, and optional inline language or
filename hints. The resolver then accepts only anchored mentions, rejects
unsupported providers without probing them, preserves percent-encoded URLs,
extracts inline code text exactly from the task text, asks for clarification on
ambiguous primary sources, and keeps invalid explicit source fields
authoritative instead of replacing them with a fallback URL or pasted snippet
from the question. GitHub issue, pull-request, and discussion URLs are
supported only as repository identity clues when the intake role marks them as
reference-only for repository-level analysis; target-content thread analysis
remains unsupported.

GitHub repository, tree, and blob scopes resolve through a managed clone.
GitHub raw-file URLs resolve through a managed single-file download under the
configured workspace. Raw download metadata uses `storage_kind:
"managed_download"` and a `raw-sha256:<hash>` content identity because the raw
HTTP response does not expose a Git commit. Existing matching managed downloads
are reused and not auto-refreshed.

Inline source bundles resolve through managed files under
`inline_sources/<bundle-id>` inside the configured workspace. Inline metadata
uses `provider: "inline"`, `storage_kind: "managed_inline_bundle"`, and an
`inline-sha256:<hash>` content identity. One inline fragment scopes to one
generated or safely hinted file. Multiple fragments scope to the managed bundle
directory. The internal manifest records hashes and hints, not full source text.

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
write patches, run project commands, install packages, call web agents, execute
inline code, or validate program output.
