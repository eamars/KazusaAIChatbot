# Retained RAG3 Evidence Boundary

kazusa_ai_chatbot.rag remains a compatibility-free package boundary for
retained RAG3 evidence helpers. RAG3/local-context searches conversation
history, memories, people, calendar context, and approved web evidence; it
returns bounded provenance-bearing evidence to cognition. Cognition owns
stance, character judgment, and response goals. Dialog owns final wording.

The live RAG3 path and its prewarm/cache owner remain outside the DSH task
session. DSH semantic tools may retrieve the same storage-independent
evidence during a task, but they do not replace the normal chat evidence
route, persona judgment, consolidation, or scheduler ownership.

Production code must import the named RAG3/local-context boundaries and
public database facades. It must not create a second supervisor graph,
deterministic user-input classifier, or task-specific evidence cache.

RAG3 tests cover bounded retrieval, provenance, time context, memory
lifecycle, prewarm behavior, and cognition projection. Run them with
venv\Scripts\python; live LLM cases run individually with saved output
inspection.
