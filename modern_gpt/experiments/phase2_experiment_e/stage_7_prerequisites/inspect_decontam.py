import json

data = json.load(open("modern_gpt/experiments/phase2_experiment_e/stage_7_prerequisites/decontamination_audit.json"))
print(f"Total violations count: {data['violating_records_count']}")
print(f"Violations by gate: {data['violations_by_gate']}")

# Group violations by candidate_id
by_cand = {}
for v in data["violations"]:
    cid = v["candidate_id"]
    if cid not in by_cand:
        by_cand[cid] = []
    by_cand[cid].append(v)

print(f"\nTotal violating candidates: {len(by_cand)}")
for cid, vlist in list(by_cand.items())[:15]:
    v = vlist[0]
    print(f"Candidate: {cid} | Gate: {v['gate_failed']} | Match: {v['matched_benchmark_id']}")
    print(f"  Details: {v['details']}")
    print(f"  Bench prompt: {v['matched_benchmark_prompt'][:80]}...\n")
