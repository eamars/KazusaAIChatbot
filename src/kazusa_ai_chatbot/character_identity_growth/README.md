# Character Identity Growth

This package owns the character-generic semantic identity contract. It defines
the complete effective identity snapshot, typed replacement paths, root
evidence validation, candidate lifecycle rules, and redacted health
precedence.

The package is a pure domain boundary:

- semantic proposal and review stages decide identity meaning, authorship,
  coherence, global applicability, contradiction, and privacy-safe
  abstraction;
- deterministic code validates closed structures, paths, types, bounds,
  repository root provenance, cadence inputs, and lifecycle transitions;
- raw MongoDB access belongs to `db.character_identity_growth`, which owns
  exactly the revision, candidate, and sanitized run collections;
- user facts, relationship state, scoped residue, cognition state, permissions,
  delivery policy, and raw transcript content remain outside identity.

An effective identity is a full snapshot. Applying patches creates a new
validated snapshot and exact diff while leaving the earlier snapshot unchanged.
Numeric LLM choices use the declared semantic bands before deterministic mapping
to unit-interval values.

Evidence counts by repository-owned `root_episode_id`. Direct and
reflection-derived representations of one root collapse into one cadence item;
reflection run IDs enrich audit lineage and never add corroboration.

Revision zero comes only from a complete canonical profile. Mongo persistence
uses immutable full snapshots, unique repository-root claims, a max-revision
reader, transaction-required promotion, and revisioned operator reset.
The service never auto-seeds: an operator must create revision zero with
`python -m scripts.load_character_profile <profile.json>` before startup.
Missing identity fails before intake and before operational state is created.

Proposal and independent review are separate background semantic stages on the
consolidation model route. Their human payload contains a band-projected
current identity, prompt-safe evidence cards, at most eight redacted current
candidates, numeric one-based row indices, and the closed allowed path list.
Each stage gets at most three complete-output attempts through the canonical
JSON parser.
Prompt overflow removes optional older candidates before failing; it never
truncates current identity or evidence.

The model-facing proposal and review objects use the V2 wire boundary: they do
not emit schema metadata, reason codes, typed replacement keys, or accepted
patch copies. Proposal patches use one `path` plus one typed `replacement`, and
provenance uses visible numeric row indices. The deterministic boundary maps
validated V2 output into the existing internal V1 decision shapes, derives
reason and change-kind fields, and copies accepted proposal patches only after
independent review validation. Internal V1 fields remain private to policy,
persistence, and downstream domain consumers.

Deterministic policy validates paths, exact review/proposal patch equality,
repository provenance, privacy-risk decisions, root/date cadence, daily caps,
stale-candidate rebasing, and fresh-root reversal thresholds. Semantic meaning,
character authorship, durability, contradiction, and safe abstraction remain
owned by the two LLM stages. The package remains free of raw database access,
character names, and personality-specific logic.

Identity is global to the character, not to one conversation partner. A close
relationship can therefore support growth in the character's own durable
capacity for love, trust, care, or reciprocal vulnerability. The relationship
target, user-specific facts, private promises, exact utterances, and intimate
details remain scoped evidence and are forbidden from the global patch. This
allows private conversation to shape later group behavior, and group
experience to shape later private behavior, only through a redacted
character-owned abstraction.

The latest revision is resolved once per cognition episode. Bounded partitions
feed moral/identity, event/agency, relationship/social, goal/threat/outcome,
existential/drive, and goal-cognition owners; `goal_cognition` includes core,
personality, boundaries, and self-image. Separate bounded projections feed
naming, text expression, and visual characterization. The first eligible
episode after promotion atomically records a sanitized consumption receipt on
the promotion run. Previous revisions remain available for review only.
