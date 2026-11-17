import json

p = "artifacts/reports/modern_gpts_100m_forensic_contamination_analysis.json"
data = json.load(open(p, "r", encoding="utf-8"))

unresolved_examples = []
for s_key, exs in data.get("representative_examples", {}).items():
    for ex in exs:
        if ex.get("classification") == "UNRESOLVED":
            unresolved_examples.append(ex)

print(f"Total representative unresolved captured: {len(unresolved_examples)}")
for i, ex in enumerate(unresolved_examples[:15]):
    print(f"[{i+1}] Suite: {ex['suite_key']} ({ex['category']}) | Record: {ex['record_id']}")
    print(f"    13-gram: {repr(ex['decoded_13gram'])}")
    print(f"    Context: {repr(ex['corpus_context'])}")
    print()

