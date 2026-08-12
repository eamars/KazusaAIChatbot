---
name: chinese-translation
description: Translate, localize, redraft, and review English or mixed-language material into native, context-aware Simplified Chinese. Use for technical documentation, architecture references, product copy, UI text, prompts, contracts, code-adjacent prose, and existing Chinese translations that read like literal machine translation or contain terminology drift. Prioritize repository and document context over word-for-word mappings while preserving meaning, contractual identifiers, and source structure.
---

# Chinese Translation

## Mission

Reconstruct the source meaning in natural Simplified Chinese. Treat translation
as a context and discourse task, not as a sentence-by-sentence substitution
exercise.

Prioritize decisions in this order:

1. Authoritative source and local context.
2. Semantic role, scope, audience, and document function.
3. Native Chinese phrasing and mainland technical register.
4. Terminology consistency within the document and repository.
5. Markdown, code, diagram, and contract preservation.

Use the mapping reference only after context lookup. A mapping is a candidate,
not an authority; let the surrounding concept, grammatical role, and local
vocabulary override it when evidence supports a better rendering.

## Multi-pass requirement

Do not translate a non-trivial document in one pass. Apply context to the full
artifact before treating any section as final.

Use at least these passes for a technical, long, or terminology-heavy
document:

1. **Context map:** identify the source structure, audience, domain, ownership
   boundaries, recurring concepts, and ambiguous terms. Start one shared term
   ledger for the entire document.
2. **Semantic draft:** translate every coherent section while preserving source
   meaning and marking low-confidence decisions. For long documents, work in
   sections, but keep the same ledger and do not finalize local wording in
   isolation.
3. **Native rewrite:** read the complete Chinese draft without the source.
   Repair discourse flow, sentence structure, register, repetition, and
   collocations. Reconcile recurring terms across section boundaries.
4. **Whole-document fidelity and artifact audit:** compare the complete target
   against the source for omissions, ownership, causality, negation, modality,
   lifecycle states, terminology distinctions, and structural integrity.

For short, low-risk text, combine the context map with the semantic draft only
when the meaning is unambiguous; still perform a Chinese-first rewrite and a
fidelity check. Treat the minimum as two passes, and use three or more whenever
context can change the translation of a term or sentence.

## Workflow

### 1. Establish authority and audience

- Identify the source language, target variant, document type, intended reader,
  and purpose: translation, localization, revision, or quality review.
- Identify the source of truth. Treat the authoritative source, schema, ICD,
  API contract, or approved terminology as higher priority than an existing
  translation.
- Separate contractual text from human-readable text. Field names, enum values,
  paths, handles, identifiers, product names, and code spans may require exact
  preservation; prose and diagram labels usually require natural rewriting.
- For repository work, inspect the target file, its source counterpart, nearby
  documentation, relevant module READMEs, and existing translations before
  editing. Use `rg` or `git grep` to find local usage of disputed terms.

### 2. Look up context before choosing words

For every ambiguous or domain-loaded term, inspect context in this order:

1. The whole sentence and its grammatical role.
2. The surrounding paragraph and section purpose.
3. Related diagrams, tables, field definitions, and examples.
4. The source implementation, schema, ICD, or code usage.
5. Existing Chinese usage in the same repository or product.
6. The reference glossary in `references/context-and-glossary.md`.

Record a temporary term ledger for non-obvious choices:

| Source term | Local meaning | Chinese candidate | Evidence | Confidence |
| --- | --- | --- | --- | --- |
| `term` | What it means here, not in isolation | Candidate wording | File, section, or code usage | high/medium/low |

Do not resolve an ambiguous term from a dictionary definition alone. Search for
its use as a noun, verb, label, field, or architectural boundary. A term can
legitimately have different Chinese renderings in different contexts.

### 3. Interpret the semantic unit

Before rewriting, identify:

- who performs each action and who owns each state;
- modality, negation, uncertainty, aspect, and causal direction;
- whether a phrase names an object, an operation, a permission, a result, or a
  boundary;
- whether the sentence is defining, contrasting, constraining, explaining, or
  narrating;
- which distinctions must remain visible to the reader.

Pay special attention to English nominalizations, passive constructions,
possessives, and causatives. Convert them into Chinese subject–predicate or
topic–comment structure instead of preserving their surface order.

### 4. Rewrite as native Simplified Chinese

- Prefer clear Chinese topic and subject order; omit repeated subjects only
  when the referent remains unambiguous.
- Turn responsibility calques such as “X owns Y” into `X 负责 Y` or
  `X 承担 Y` when the source means responsibility rather than possession.
- Render causatives such as “must not make the model infer” as `不能让模型推断`
  rather than `不能要求模型推断`.
- Restore Chinese measure words, 的-structures, connective logic, and natural
  verb–object collocations.
- Replace English noun stacks with explicit Chinese relations. Make the
  subject, scope, and consequence visible when Chinese syntax requires it.
- Prefer mainland Simplified Chinese technical register. Avoid regional,
  financial, legal, biological, literary, or auction metaphors unless the
  source genuinely means them.
- Preserve the source’s level of formality. Make architecture prose precise and
  readable without turning it into marketing copy or literary prose.

### 5. Apply terminology with context priority

- Use repository terminology when it is established and semantically correct.
- Keep one concept stable within a document, but do not force one Chinese word
  onto distinct source concepts merely to satisfy a one-to-one table.
- Distinguish near-neighbor concepts before polishing style. Examples include
  capability vs. affordance, affinity vs. closeness vs. intimacy, an object vs.
  a lookup operation, and an authorization vs. an authorization credential.
- Read `references/context-and-glossary.md` when the document uses any of its
  flagged terms. Treat its suggestions as hypotheses to confirm against local
  context.
- Preserve contractual English identifiers in code spans and schema-bearing
  text. Translate their human explanation separately.

### 6. Preserve artifact structure

For Markdown and code-adjacent documents:

- Keep heading hierarchy, section order, tables, bullets, examples, and fenced
  block count aligned with the source unless the user explicitly requests a
  structural change.
- Preserve link targets, code spans, field names, enum values, paths, handles,
  and product names byte-for-byte when they are contractual.
- Preserve Mermaid or other diagram IDs, arrows, participants, and message
  direction. Translate human-readable labels only. Keep syntax-required ASCII
  punctuation inside diagram declarations.
- Do not add an explanatory glossary, architecture claim, compatibility alias,
  or translator note unless the user requests it or the target artifact already
  defines that convention.
- Keep unrelated worktree changes intact and limit the diff to the requested
  translation artifact.

### 7. Review in separate passes

Run these passes in order:

1. **Chinese-first fluency pass:** read the target without looking at the
   source. Remove literal syntax, redundant pronouns, awkward nominalizations,
   ambiguous modifiers, and unnatural collocations.
2. **Context pass:** revisit every low-confidence term and compare it with local
   source usage, adjacent sections, schemas, and diagrams.
3. **Bilingual fidelity pass:** compare each paragraph and table row with the
   source. Restore omitted constraints, negations, ownership, causality,
   uncertainty, and technical distinctions.
4. **Terminology pass:** search for inconsistent renderings, collisions, and
   accidental translations of identifiers.
5. **Artifact pass:** run `git diff --check`; compare headings, links, code
   spans, fences, and diagram topology when structure matters.

Report unresolved ambiguity instead of silently inventing a semantic decision.

## Quality bar

Accept a translation only when a native reader can understand it without
mentally reconstructing English syntax, and a technical reviewer can map every
important decision back to the authoritative source and local context.

Reject a draft that is merely fluent but changes ownership, permission,
causality, scope, modality, lifecycle state, or the distinction between an
observation and a judgment.

## Reference

Read [context-and-glossary.md](references/context-and-glossary.md) for the
context-lookup checklist and examples derived from the architecture translation
review. Use it as a secondary aid after repository lookup, never as a blind
replacement table.
