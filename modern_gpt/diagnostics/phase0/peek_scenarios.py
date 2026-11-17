import json

with open("modern_gpt/reports/phase0_forensic_audit/generation_results.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        rec = json.loads(line)
        if rec.get("test_type") == "multiturn_scenario":
            print(f"\n=== Line {idx}: [{rec['checkpoint']}] {rec['decoding_config']} | Scenario: {rec['prompt_id']} ===")
            for h in rec.get("scenario_history", []):
                print(f"  Turn {h['turn']} User: {repr(h['user'])}")
                print(f"  Turn {h['turn']} Asst: {repr(h['assistant'][:120])}")
                print(f"       Stop: {h['stop_reason']}")

