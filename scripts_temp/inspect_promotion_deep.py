"""Deep inspect reflection promotion artifacts."""
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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("=" * 72)
    print(f"CASE: {data.get('case_id')}")
    print("=" * 72)
    # Show all keys at top level
    for key in sorted(data.keys()):
        if key == "payload":
            payload = data[key]
            # Show payload keys
            print(f"  payload keys: {sorted(payload.keys())}")
            if "raw_output" in payload:
                print(f"  payload.raw_output:\n{payload['raw_output'][:1200]}")
            if "promote_decisions" in payload:
                print(f"  payload.promote_decisions:")
                print(json.dumps(payload["promote_decisions"], indent=2, ensure_ascii=False)[:1200])
            if "parsed_output" in payload:
                print(f"  payload.parsed_output:")
                print(json.dumps(payload["parsed_output"], indent=2, ensure_ascii=False)[:1200])
            if "response" in payload:
                print(f"  payload.response:")
                resp = payload["response"]
                if isinstance(resp, str):
                    print(resp[:1200])
                else:
                    print(json.dumps(resp, indent=2, ensure_ascii=False)[:1200])
        else:
            val = data[key]
            if isinstance(val, (str, int, float, bool, type(None))):
                print(f"  {key}: {val}")
    print()
