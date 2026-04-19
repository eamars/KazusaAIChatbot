# Depth Classifier Updater — Quick Reference

## Quick Start

### One-Command Update (Recommended)
```bash
cd c:\workspace\kazusa_ai_chatbot
python .agents/skills/depth-classifier-updater/update_classifier.py
```

### With Custom Sample Size
```bash
python .agents/skills/depth-classifier-updater/update_classifier.py --sample-size 1000
```

### Dry-Run (No File Updates)
```bash
python .agents/skills/depth-classifier-updater/update_classifier.py --output-only
```

## Process Overview

```
Query MongoDB           Extract Keywords        Classify Messages
─────────────────  →  ──────────────────  →  ─────────────────
conversation_history   Phrases 3-6 words      SHALLOW vs DEEP
500+ messages          Count frequencies      Signal matching


Deduplicate/Balance    Update Classifier      Run Tests
──────────────────  →  ─────────────────  →  ──────────
45% EN / 55% ZH        depth_classifier.py    Verify accuracy
Remove substrings      Generate keyword lists Test coverage
```

## Key Metrics to Track

After each update, record:

| Metric | Target | Tool |
|--------|--------|------|
| Sample size | 300-1000 messages | `--sample-size` arg |
| Language balance | 45-55% EN/ZH | Output log |
| Keyword count SHALLOW | 25-40 | Output log |
| Keyword count DEEP | 30-50 | Output log |
| Test accuracy | >85% | `python -m src.kazusa_ai_chatbot.rag.depth_classifier` |
| Centroid compute time | <100ms | Test output |
| Affinity override | Always DEEP when <400 | Test output |

## Signal Reference

### SHALLOW Signals (✓ Use cached results)
```
Greetings: hello, 你好, good morning, 早上好
Direct Q's: what time, how are you, 现在几点
Yes/No: yes, no, okay, 好不好, 是吗
Social: thanks, 谢谢, you're welcome, 客气
Basic facts: what color, do you like, 什么颜色
```

### DEEP Signals (✗ Need full search)
```
Past references: remember, 你还记得, i told you
Promises: promised, 你答应, you said
Temporal: compared to, 以前, last time, 上次
Emotional: feel, love, 喜欢, how do you feel
Personal: my hobby, 我的爱好, what about me
Contradictions: but you, 但你, you said before
```

## Configuration

### Edit Signal Weights (if needed)

File: `.agents/skills/depth-classifier-updater/update_classifier.py`

```python
SHALLOW_SIGNALS = {
    "hello": 1.0,        # Weight 1.0 = strong signal
    "what time": 0.9,    # 0.9 = moderate signal
    "is there": 0.6,     # 0.6 = weak signal
    # Add more...
}

DEEP_SIGNALS = {
    "remember": 1.0,
    "i told you": 1.0,
    "you promised": 1.0,
    # Add more...
}
```

### Tune Parameters

```python
def deduplicate_and_balance(
    phrases,
    min_count=3,           # Phrases appearing 3+ times
    target_count=35,       # Total keywords per category
):
    # Lower min_count = more keywords
    # Raise min_count = only popular keywords
```

## Integration Example

Add to CI/CD pipeline (cron job):

```bash
# Weekly keyword update
0 2 * * 0 cd /workspace/kazusa_ai_chatbot && \
  python .agents/skills/depth-classifier-updater/update_classifier.py && \
  python -m pytest tests/test_depth_classifier.py && \
  git add src/kazusa_ai_chatbot/rag/depth_classifier.py && \
  git commit -m "Weekly: Update depth classifier keywords"
```

## Troubleshooting

**Issue: "Failed to import kazusa_ai_chatbot.db"**
```bash
# Solution: Ensure venv is activated
source venv/Scripts/activate  # Windows
python -m pip install -e .
```

**Issue: "Failed to query conversation_history"**
```bash
# Solution: Verify MongoDB connection
mongo mongodb://localhost:27027/kazusa_bot_core
db.conversation_history.find().limit(1)
```

**Issue: "Classifier accuracy is still low"**
```bash
# Solution: Check keyword distribution
python update_classifier.py --output-only

# If many SHALLOW keywords are too generic:
# → Remove "what", "你", "的" from SHALLOW
# → Add more specific phrases

# If many DEEP keywords are false positives:
# → Review signal weights in DEEP_SIGNALS
# → Increase min_count from 3 to 5
```

**Issue: "Language balance is off"**
```bash
# Solution: Manually adjust extracted keywords
# Edit output before updating file:
python update_classifier.py --output-only | grep -E "^    \"" > /tmp/keywords.txt
# Edit /tmp/keywords.txt manually
# Update classifier_file directly (or run with manual fix)
```

## File Structure

```
.agents/skills/depth-classifier-updater/
├── SKILL.md                      # This comprehensive guide
├── QUICK_REFERENCE.md            # You are here
└── update_classifier.py           # Automation script
```

## Related Commands

```bash
# Test classifier after update
python -m src.kazusa_ai_chatbot.rag.depth_classifier

# Check classifier accuracy
python -m pytest tests/test_depth_classifier.py -v

# View current keywords
grep -A 50 "SHALLOW_KEYWORDS" src/kazusa_ai_chatbot/rag/depth_classifier.py

# View MongoDB sample
mongo mongodb://localhost:27027/kazusa_bot_core
db.conversation_history.find({"role": "user"}).limit(5).pretty()
```

## Best Practices

1. **Run monthly or after 500+ new messages** — Don't update too frequently
2. **Always test after update** — Run full test suite before committing
3. **Review keyword diff** — Check what changed before committing to git
4. **Maintain language balance** — Keep ~50% English, ~50% Chinese
5. **Version control** — Commit update with date and metrics in message
6. **Monitor accuracy** — Track false positive/negative rates in production
7. **Archive old keywords** — Keep git history for rollback if needed

## Metrics Dashboard (Optional)

Track this over time:

```
Date       | Sample | Shallow | Deep | Test Acc | Centroid | Notes
-----------|--------|---------|------|----------|----------|----------
2026-04-19 | 467    | 35      | 42   | 89%      | 42ms     | Initial update
2026-05-03 | 523    | 38      | 45   | 91%      | 48ms     | Added emotions
```

## Support

For issues or questions about the skill:
- Review SKILL.md for detailed workflow
- Check `.agents/MEMORY_DEDUPLICATION_STRATEGY.md` for depth semantics
- Inspect `tests/test_depth_classifier.py` for test patterns
