import json
from pathlib import Path

p = Path("artifacts/reports/modern_gpts_100m_forensic_contamination_analysis.json")
if not p.exists():
    print("Report not found.")
    exit(1)

with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 60)
print("FORENSIC CONTAMINATION SUMMARY (1 SHARD)")
print("=" * 60)
print(f"Total matching occurrences: {data['overall_metrics']['total_matching_occurrences']:,}")
print(f"Unique matching 13-grams:   {data['overall_metrics']['unique_matching_13grams']:,}")
print()
print("SUITE BREAKDOWN:")
for k, v in data["metrics_by_reference_suite"].items():
    print(f"[{k}] ({v['category']}):")
    print(f"  Ref 13-grams:    {v['unique_reference_13grams']:,}")
    print(f"  Unique Matches:  {v['unique_matching_13grams']:,} ({v['match_rate_pct']}%)")
    print(f"  Occurrences:     {v['total_matching_occurrences']:,}")

print()
print("CLASSIFICATION BREAKDOWN:")
for k, v in data["classification_breakdown"].items():
    print(f"  {k}: {v:,}")

print()
print("CRITICAL SUITE MATCH EXAMPLES:")
for suite in ["critical_held_out_100", "critical_transfer_160", "critical_checkpoint_selection"]:
    exs = data.get("representative_examples", {}).get(suite, [])
    print(f"\n--- {suite} ({len(exs)} captured examples) ---")
    for ex in exs[:5]:
        print(f"  Record: {ex['record_id']} | Field: {ex['field']} | Class: {ex['classification']}")
        print(f"  Decoded 13-gram: {repr(ex['decoded_13gram'])}")
        print(f"  Corpus Context:  {repr(ex['corpus_context'])}")
        print(f"  Reason: {ex['classification_reason']}")
        print()

