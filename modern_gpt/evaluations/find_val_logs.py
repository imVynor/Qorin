import json
from pathlib import Path

telemetry_path = Path("modern_gpt/reports/phase1c_pretraining/training_telemetry.jsonl")

val_records = []
with open(telemetry_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "val_loss" in data or "eval" in data or "validation" in data:
            val_records.append(data)

print(f"Found {len(val_records)} validation records in training_telemetry.jsonl:")
for r in val_records:
    print(r)
