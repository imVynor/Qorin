import json

with open("modern_gpt/diagnostics/phase0/qwen_comparison_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for r in data:
    if "prompt" in r:
        print(f"Qwen on [{r['suite']}] '{r['prompt']}':")
        print("  ->", repr(r["response"][:150]))
    else:
        print(f"\nQwen Scenario [{r['suite']}]:")
        for h in r["scenario_history"]:
            print(f"  Turn {h['turn']} User: {repr(h['user'])}")
            print(f"  Turn {h['turn']} Asst: {repr(h['assistant'][:120])}")

