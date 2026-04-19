# Depth Classifier Updater Skill

**Purpose:** Automatically extract and update SHALLOW/DEEP keywords in the depth classifier based on real conversation data from MongoDB.

**Status:** ✅ Ready for evaluation and testing

**Compatibility:** Requires Python 3.10+, MongoDB connection, `kazusa_ai_chatbot` package installed

## Files in This Skill

| File | Purpose |
|------|---------|
| **SKILL.md** | Comprehensive workflow, best practices, implementation details |
| **QUICK_REFERENCE.md** | Quick start guide and troubleshooting |
| **README.md** | You are here |
| **evals/evals.json** | Test cases for skill validation (3 scenarios) |
| **update_classifier.py** | Automation script (bundled resource) |

## Quick Start

### 1. One-Command Update
```bash
cd c:\workspace\kazusa_ai_chatbot
python .agents/skills/depth-classifier-updater/update_classifier.py
```

### 2. Review Output
```
✅ Extracted keywords:
   SHALLOW: 35 keywords
   DEEP: 42 keywords
   Sample size: 467 messages
```

### 3. Test Classifier
```bash
python -m src.kazusa_ai_chatbot.rag.depth_classifier
```

### 4. Commit
```bash
git add src/kazusa_ai_chatbot/rag/depth_classifier.py
git commit -m "chore: Update depth classifier keywords (467 samples, 89% accuracy)"
```

## Why This Skill?

The depth classifier determines whether user queries require SHALLOW (cache-only) or DEEP (full search) processing. It learns from hardcoded keyword lists that become outdated as user interaction patterns evolve.

**The problem:**
- **Hardcoded keywords drift over time** — Real users say things differently than anticipated
- **Different bots have different patterns** — What's SHALLOW for one bot is DEEP for another
- **Language balance can shift** — English users might become 70% of traffic
- **New interaction patterns emerge** — Slang, memes, seasonal topics

**This skill keeps the classifier aligned with reality** by:
1. Extracting 300-1000 recent messages from MongoDB
2. Analyzing them with signal-based classification
3. Identifying frequently-used phrases
4. Deduplicating and balancing English/Chinese
5. Automatically updating the classifier file
6. Running validation tests

## When to Use This Skill

- **Monthly maintenance** — Keep keywords fresh
- **After deploying new features** — User interaction patterns change
- **When classifier accuracy drops** — Too many false positives/negatives
- **Quarterly refresh** — Major keyword overhaul with larger dataset
- **Before major releases** — Ensure classifier reflects current usage

## How It Works

```
MongoDB                    Classify                    Extract
conversation_history   →   SHALLOW/DEEP    →    3-6 word phrases
(500+ messages)           (signal-based)        (frequency counted)
                                ↓
                          Deduplicate
                          & Balance
                          (EN/ZH split)
                                ↓
                          Update File
                        depth_classifier.py
                                ↓
                          Run Tests
                          & Validate
```

## Evaluation Structure

This skill includes test cases for proper validation per skill-creator guidelines.

**Test cases** (see `evals/evals.json`):
1. **Full update**: Extract 500 messages, update classifier, run tests
2. **Dry-run**: Extract 300 messages, show results, DON'T update files
3. **Custom sample size**: Extract 800 messages with `--sample-size` flag

**Assertions for each test:**
- ✅ File modifications (or lack thereof)
- ✅ Language balance: 45-55% English/Chinese
- ✅ Keyword counts: 25-50 SHALLOW, 30-50 DEEP
- ✅ Test accuracy: ≥85%
- ✅ Proper error handling

## Key Concepts

### SHALLOW Queries
Simple, factual, no memory needed:
- Greetings: "Hello", "你好", "Good morning"
- Direct questions: "What time is it?", "现在几点?"
- Yes/no questions: "Yes", "是吗?"
- Context-free preferences: "Do you like red?", "你喜欢什么颜色?"

### DEEP Queries
Complex, need historical context:
- Past references: "Remember when I...", "你还记得吗"
- Promises: "You promised", "你答应过"
- Temporal reasoning: "Compared to before", "上次"
- Emotional: "How do you feel?", "你对我..."
- Personal facts: "What did I tell you?", "我的爱好"

## Configuration

Edit signal weights in `update_classifier.py` if needed:

```python
SHALLOW_SIGNALS = {
    "hello": 1.0,          # Strong signal
    "what time": 0.9,      # Moderate signal
    "is there": 0.6,       # Weak signal
}

DEEP_SIGNALS = {
    "remember": 1.0,
    "you promised": 1.0,
}
```

## Integration

Add to cron job for weekly updates:

```bash
0 2 * * 0 cd /workspace/kazusa_ai_chatbot && \
  python .agents/skills/depth-classifier-updater/update_classifier.py && \
  python -m pytest tests/test_depth_classifier.py
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Too many false SHALLOW | Add more emotional/temporal keywords to DEEP |
| Too many false DEEP | Simplify keywords, remove ambiguous phrases |
| Unbalanced languages | Add missing EN/ZH equivalents |
| Low test accuracy | Increase sample size to 500+ |
| MongoDB connection error | Verify connection string in .env |

See **QUICK_REFERENCE.md** for detailed troubleshooting.

## Next Steps

1. **Read SKILL.md** — Understand full workflow and best practices
2. **Read QUICK_REFERENCE.md** — Quick start and configuration
3. **Review evals/evals.json** — Understand test cases
4. **Run update_classifier.py** — Execute first update
5. **Review classifier tests** — Verify accuracy with `pytest tests/test_depth_classifier.py`
6. **Schedule monthly runs** — Add to cron or CI/CD pipeline

## Related Documentation

- **Classifier implementation**: `src/kazusa_ai_chatbot/rag/depth_classifier.py`
- **Classifier tests**: `tests/test_depth_classifier.py`
- **Memory strategy**: `.agents/MEMORY_DEDUPLICATION_STRATEGY.md`
- **Cache system**: `src/kazusa_ai_chatbot/rag/cache.py`

## Files Modified by This Skill

- `src/kazusa_ai_chatbot/rag/depth_classifier.py` — SHALLOW/DEEP keywords lists
- (No other files are modified)

## Performance

- **Extraction**: ~5-10 seconds (depends on MongoDB query time)
- **Keyword processing**: ~1 second
- **File update**: <100ms
- **Total**: ~10-15 seconds

---

**Created:** April 19, 2026  
**Last Updated:** April 19, 2026  
**Skill Version:** 1.0  
**Evaluated:** Pending formal evaluation via skill-creator workflow
