import json

p = "artifacts/reports/modern_gpts_100m_forensic_contamination_analysis.json"
data = json.load(open(p, "r", encoding="utf-8"))

for s in ["critical_held_out_100", "critical_transfer_160"]:
    exs = data.get("representative_examples", {}).get(s, [])
    print(f"\n========================================================")
    print(f"=== {s}: {len(exs)} captured examples ===")
    print(f"========================================================")
    seen_records = set()
    for ex in exs:
        rec_id = ex["record_id"]
        if rec_id not in seen_records:
            seen_records.add(rec_id)
            print(f"Record: {rec_id} | Field: {ex['field']} | Class: {ex['classification']}")
            print(f"   Decoded: {repr(ex['decoded_13gram'])}")
            print(f"   Context: {repr(ex['corpus_context'])}")
            print(f"   Reason:  {ex['classification_reason']}")
            print()
