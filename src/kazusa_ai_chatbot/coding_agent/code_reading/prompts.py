"""Prompt templates reserved for future LLM-backed code reading."""

EVIDENCE_ONLY_SYNTHESIS_POLICY = (
    "Synthesize only from bounded evidence rows. Do not include local checkout "
    "paths, workspace roots, cache keys, raw command output, or full source "
    "files in prompts."
)
