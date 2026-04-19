---
name: depth-classifier-updater
description: Automatically extract and update the SHALLOW/DEEP keyword lists that drive query classification in the depth classifier, keeping it aligned with real conversation patterns. Use this skill whenever you're asked to maintain or improve the classifier, after deploying new features that change how users interact, when classifier accuracy drops (too many false SHALLOW/DEEP misclassifications), for regular maintenance (monthly), before major releases, or whenever you notice the hardcoded keywords no longer match actual usage patterns. The skill extracts real conversation data from MongoDB, classifies queries, deduplicates phrases, balances English/Chinese, updates the classifier file automatically, and validates the changes with tests.
---

# Depth Classifier Keyword Updater

## Quick Overview
This skill keeps the depth classifier synchronized with reality. Over time, hardcoded keywords drift as user behavior evolves. This skill uses actual conversation history to regenerate the keyword lists, ensuring the classifier makes better routing decisions (SHALLOW = cached results OK, DEEP = need full context search).

## Why This Matters

The depth classifier decides how to answer each query:
- **SHALLOW queries** → Use memory cache only (fast, deterministic)
- **DEEP queries** → Full context search with LLM (slower, more accurate)

The problem: Classifiers learn from fixed keyword lists that become outdated. As users interact, they develop new phrasings, adopt new topics, and their communication style evolves. A classifier that worked 3 months ago may now misclassify 20% of queries, wasting compute on unnecessary searches or returning stale cached results.

This skill continuously recalibrates the classifier to match actual usage, improving accuracy and efficiency.

## When to Use

- **Monthly maintenance** — Refresh keywords after 500+ new conversations
- **After feature deployments** — User interaction patterns often change
- **When classifier accuracy drops** — Too many false positives/negatives
- **Before major releases** — Ensure classifier reflects current usage
- **When bilingual balance shifts** — Ensure English/Chinese coverage stays even

## Basic Workflow

The skill automates these steps:

1. **Extract** → Query MongoDB for 300-1000 recent user messages
2. **Classify** → Label each message SHALLOW or DEEP using signal matching
3. **Extract phrases** → Find frequently-used 3-6 word phrases (3+ occurrences)
4. **Deduplicate** → Remove substring duplicates, merge synonyms
5. **Balance** → Ensure 45-55% English / 45-55% Chinese split
6. **Update** → Replace keyword lists in `depth_classifier.py`
7. **Test** → Validate classifier accuracy is ≥85%

A bundled script handles all this: `scripts/update_classifier.py`

```bash
# One-command update (recommended)
python .agents/skills/depth-classifier-updater/update_classifier.py

# Dry-run (show proposed keywords, don't update file)
python .agents/skills/depth-classifier-updater/update_classifier.py --output-only

# Custom sample size
python .agents/skills/depth-classifier-updater/update_classifier.py --sample-size 1000
```

## Key Concepts
**Characteristics:**
- Simple factual questions that don't require memory
- Greetings and social pleasantries
- Direct yes/no questions
- Basic preference checks
- Current state queries (time, weather)

**Examples:**
- "Hello", "你好", "Good morning"
- "What time is it?", "现在几点?"
- "Do you like red?", "你喜欢什么颜色?"
- "Are you here?", "你在么?"

### DEEP Queries
**Characteristics:**
- References to past conversations or events
- Questions about promises or commitments
- Temporal reasoning ("remember when", "last time", "compared to before")
- Emotional context or relationship changes
- Contradictions ("but you said", "you promised")
- Personal facts retrieval ("what did I tell you", "do you remember my")

**Examples:**
- "Remember when I visited Japan?", "我之前告诉你"
- "You promised to help me", "你答应过"
- "Why did you say that?", "为什么你总是"
- "Do you remember my hobby?", "你还记得吗"

## Workflow

### Step 1: Extract Conversation Data
```python
# Query MongoDB for user messages
db.kazusa_bot_core.conversation_history.find(
    {"role": "user"},
    {"content": 1, "_id": 0}
).limit(500)  # Sample recent conversations
```

**Best Practices:**
- Sample 300-1000 recent messages (not too old, not too small)
- Focus on non-test messages (filter out system prompts, eval attempts)
- Weight recent data more heavily (last 2 weeks)
- Include both English and Chinese messages equally

### Step 2: Analyze Message Patterns
For each message, classify into:
- **SHALLOW**: Simple, direct, no historical context needed
- **DEEP**: References past, reasoning, emotional content
- **AMBIGUOUS**: Could be either (default to DEEP for safety)

**Classification Signals:**

| Signal | Type | Examples |
|--------|------|----------|
| Greeting | SHALLOW | hello, 你好, good morning |
| Direct question | SHALLOW | what time, how are you |
| Yes/no | SHALLOW | yes, okay, 是吗, 好不好 |
| Past reference | DEEP | remember, visited, 以前, 上次 |
| Promise/obligation | DEEP | promised, told you, 答应, 告诉你 |
| Reasoning | DEEP | why, compared, but you said, 为什么 |
| Emotional | DEEP | feel, love, hate, 喜欢, 讨厌 |
| Personal fact | DEEP | my hobby, remember me, 我的爱好 |

### Step 3: Extract Frequent Phrases (3-6 words)
- Count phrase frequency (phrases appearing 3+ times are significant)
- Normalize: lowercase, remove punctuation
- Weight by recency
- Ensure 50/50 English/Chinese split

**Example Output:**
```
SHALLOW_CANDIDATES:
- "hello" (12x, English)
- "你好" (8x, Chinese)
- "what time is it" (5x, English)
- "你在么" (7x, Chinese)
- "good morning" (6x, English)

DEEP_CANDIDATES:
- "remember when" (9x, English)
- "你还记得吗" (6x, Chinese)
- "i told you" (8x, English)
- "你答应过" (5x, Chinese)
- "compared to before" (4x, English)
```

### Step 4: Validate and Deduplicate
- Remove phrases that are substrings of other phrases
- Remove overly general phrases ("what", "是吗", "你好吗")
- Merge synonyms ("hi" + "hello" → keep "hello", add "hi")
- Verify language balance (English ≈ Chinese)

**Deduplication Rules:**
```
If "remember when" and "remember" exist:
  → Keep "remember when" (more specific)
  
If "你好" and "你好吗" exist:
  → Keep both (different depths: "你好" SHALLOW, "你好吗" could be DEEP)
  
If duplicate phrases in different languages:
  → Keep one instance from each language
```

### Step 5: Update Classifier File
Replace `SHALLOW_KEYWORDS` and `DEEP_KEYWORDS` lists in:
`src/kazusa_ai_chatbot/rag/depth_classifier.py`

**Constraints:**
- Maintain ~30-50 keywords per category (too many = slower, too few = less accurate)
- Preserve any manually-added domain-specific keywords
- Add comments showing update date and keyword count
- Keep alphabetical order within language sections

### Step 6: Test Updated Classifier
Run comprehensive tests to validate:
```bash
python -m src.kazusa_ai_chatbot.rag.depth_classifier
```

**Test Cases to Verify:**
1. ✅ Known SHALLOW queries → return SHALLOW
2. ✅ Known DEEP queries → return DEEP
3. ✅ Affinity override works (affinity < 400 → DEEP)
4. ✅ Bilingual support (English and Chinese work equally)
5. ✅ Fallback to LLM when ambiguous
6. ✅ Centroid computation completes without errors

## Implementation Script Outline

```python
# update_depth_classifier.py (pseudocode)

import asyncio
from collections import Counter
from kazusa_ai_chatbot.db import get_db

async def extract_keywords_from_conversations():
    """Extract keywords from MongoDB conversation history."""
    db = await get_db()
    
    # Query recent conversations (limit 500)
    conversations = await db.kazusa_bot_core.conversation_history.find(
        {"role": "user"},
        {"content": 1}
    ).limit(500).to_list(500)
    
    # Classify each message
    shallow_phrases = []
    deep_phrases = []
    
    for doc in conversations:
        content = doc.get("content", "").strip()
        if not content or len(content) < 3:
            continue
            
        depth = classify_message(content)  # Your custom classifier
        phrases = extract_phrases(content)  # 3-6 word phrases
        
        if depth == "SHALLOW":
            shallow_phrases.extend(phrases)
        elif depth == "DEEP":
            deep_phrases.extend(phrases)
    
    # Get most common phrases (3+ occurrences)
    shallow_top = [p for p, count in Counter(shallow_phrases).most_common(30) 
                   if count >= 3]
    deep_top = [p for p, count in Counter(deep_phrases).most_common(40) 
                if count >= 3]
    
    # Deduplicate and balance languages
    shallow_updated = deduplicate_and_balance(shallow_top)
    deep_updated = deduplicate_and_balance(deep_top)
    
    # Update classifier file
    update_classifier_file(shallow_updated, deep_updated)
    
    # Run tests
    await test_classifier()
    
    return {
        "shallow_count": len(shallow_updated),
        "deep_count": len(deep_updated),
        "status": "SUCCESS"
    }
```

## Quality Checklist

Before committing keyword updates:

- [ ] Sample size: 300+ recent messages (last 2-4 weeks)
- [ ] Language balance: 45-55% English, 45-55% Chinese
- [ ] No generic words: "what", "是吗", "你好吗" removed
- [ ] No substrings: "remember when" kept, "remember" removed if present
- [ ] Keyword count: 25-50 SHALLOW, 30-50 DEEP
- [ ] Test suite: All 6 test cases passing
- [ ] Affinity override: Works correctly (affinity < 400 → DEEP)
- [ ] Git commit: Message includes date, keyword counts, accuracy notes

## Performance Considerations

- **Embedding computation**: 60-80 keywords × 2 = 160 embeddings per update (30-50s)
- **LLM fallback cost**: Each ambiguous query calls LLM (slower but more accurate)
- **Cache invalidation**: After update, centroid cache is recalculated on first query
- **Frequency**: Run weekly for high-volume bots, monthly for low-volume

## Maintenance Tasks

**Monthly:**
- [ ] Review keyword accuracy on real conversations
- [ ] Check for new patterns (seasonal, event-driven)
- [ ] Verify language balance hasn't drifted
- [ ] Update this skill if interaction patterns change

**Quarterly:**
- [ ] Major keyword refresh with 1000+ messages
- [ ] Retrain centroids with larger dataset
- [ ] Review affinity threshold (currently 400)
- [ ] Document any domain-specific keywords

**Before Deployment:**
- [ ] Run full test suite
- [ ] Compare old vs new keyword sets
- [ ] Manual spot-check on 10-20 real queries
- [ ] Verify no regression in classifier accuracy

## Related Files

- **Classifier**: `src/kazusa_ai_chatbot/rag/depth_classifier.py`
- **Tests**: `tests/test_depth_classifier.py`
- **Config**: `.agents/MEMORY_DEDUPLICATION_STRATEGY.md` (DEEP/SHALLOW thresholds)
- **Database**: `kazusa_bot_core.conversation_history`

## Example Update Output

```
=== Depth Classifier Keyword Update ===
Date: 2026-04-19
Sample size: 467 messages (last 14 days)
Language split: 48% EN, 52% ZH

SHALLOW_KEYWORDS (35 total):
  - hello, hi, hey, good morning, good night, thanks, thank you (EN)
  - 你好, 你在么, 早上好, 晚安, 谢谢, 知道了 (ZH)
  [+29 more...]

DEEP_KEYWORDS (42 total):
  - remember when, i told you, i visited, do you remember (EN)
  - 你还记得吗, 你答应过, 我之前, 为什么 (ZH)
  [+34 more...]

Test Results:
  ✅ SHALLOW accuracy: 92%
  ✅ DEEP accuracy: 89%
  ✅ Affinity override: PASS
  ✅ Bilingual support: PASS
  ✅ Centroid computation: 42ms

Recommendation: ✅ READY FOR DEPLOYMENT
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many false SHALLOW | Add more emotional/temporal keywords to DEEP |
| Too many false DEEP | Simplify DEEP keywords, remove ambiguous phrases |
| Unbalanced languages | Manually add missing Chinese/English phrases |
| Low accuracy overall | Sample size too small; run on 500+ messages |
| Slow centroid computation | Reduce keyword count to 25-40 per category |

## See Also
- Depth classification algorithm: `src/kazusa_ai_chatbot/rag/depth_classifier.py#classify()`
- Memory strategy: `.agents/MEMORY_DEDUPLICATION_STRATEGY.md`
- Semantic cache types: `src/kazusa_ai_chatbot/rag/cache.py#DEFAULT_TTL_SECONDS`
