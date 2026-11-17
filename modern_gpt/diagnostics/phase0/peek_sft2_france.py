import json

with open("modern_gpt/reports/phase0_forensic_audit/generation_results.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx in (280, 281, 282, 283):
            rec = json.loads(line)
            print(f"=== [{rec['checkpoint']}] {rec['decoding_config']} | Prompt: {rec.get('prompt_text')} ===")
            print("Response:", repr(rec["response"][:200]))
            print("Stop reason:", rec["stop_reason"])

