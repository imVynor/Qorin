import json

with open("modern_gpt/reports/phase0_forensic_audit/generation_results.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print("Total lines:", len(lines))
    for idx in range(max(0, len(lines) - 5), len(lines)):
        rec = json.loads(lines[idx])
        print(f"Line {idx}: ckpt={rec.get('checkpoint')}, cfg={rec.get('decoding_config')}, type={rec.get('test_type')}, suite={rec.get('suite')}, prompt={repr(rec.get('prompt_text') or rec.get('prompt_id'))}")

