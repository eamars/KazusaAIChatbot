"""Inspect reflection promotion live LLM artifacts."""
import json
from pathlib import Path

base = Path("test_artifacts/llm_traces")
cases = [
    "reflection_cycle_stage1c_promotion_live_llm__normal_case.json",
    "reflection_cycle_stage1c_promotion_live_llm__privacy_rejection_case.json",
    "reflection_cycle_stage1c_promotion_live_llm__no_signal_case.json",
]

for filename in cases:
    path = base / filename
    if not path.exists():
        print(f"MISSING: {filename}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("=" * 72)
    print(f"CASE: {filename}")
    print("=" * 72)
    # Show the promote_decisions or key fields
    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, (str, int, float, bool, type(None))):
            print(f"  {key}: {val}")
        elif isinstance(val, list) and len(val) <= 3:
            print(f"  {key}: {json.dumps(val, indent=4, ensure_ascii=False)[:500]}")
        elif isinstance(val, dict):
            print(f"  {key}: {json.dumps(val, indent=4, ensure_ascii=False)[:500]}")
        else:
            print(f"  {key}: ({type(val).__name__}, len={len(val)})")
    print()
