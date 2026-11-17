import json

with open("modern_gpt/reports/phase0_forensic_audit/generation_results.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx == 248:
            rec = json.loads(line)
            print(f"=== [{rec['checkpoint']}] {rec['decoding_config']} | Scenario: {rec['prompt_id']} ===")
            for h in rec.get("scenario_history", []):
                print(f"  Turn {h['turn']} User: {repr(h['user'])}")
                print(f"  Turn {h['turn']} Asst: {repr(h['assistant'])}")

