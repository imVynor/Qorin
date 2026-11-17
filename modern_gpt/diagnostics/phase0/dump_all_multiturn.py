import json

with open("modern_gpt/reports/phase0_forensic_audit/generation_results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("test_type") == "multiturn_scenario":
            ckpt = rec["checkpoint"]
            dec = rec["decoding_config"]
            scen = rec["prompt_id"]
            print(f"\n==================== [{ckpt}] {dec} | {scen} ====================")
            for h in rec["scenario_history"]:
                print(f"Turn {h['turn']} User: {repr(h['user'])}")
                print(f"Turn {h['turn']} Asst: {repr(h['assistant'][:150])}")
                print(f"       Stop reason: {h['stop_reason']}")

