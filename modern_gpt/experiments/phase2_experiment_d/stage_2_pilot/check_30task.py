import json
import glob

print("--- Listing files in stage_2_pilot/logs ---")
for fpath in glob.glob("modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/logs/*"):
    print(fpath)

print("\n--- Reading detailed_30task_results.json ---")
with open('modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/logs/detailed_30task_results.json') as f:
    d = json.load(f)
print("Keys:", d.keys())
print("exp_d[0]:", json.dumps(d['exp_d'][0], indent=2))

print("\n--- Reading pilot_evaluation_results.json ---")
try:
    with open('modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/logs/pilot_evaluation_results.json') as f:
        p = json.load(f)
    print("pilot_evaluation_results keys:", p.keys() if isinstance(p, dict) else len(p))
    if isinstance(p, dict):
        for k in list(p.keys())[:5]:
            print(k, type(p[k]))
except Exception as e:
    print("Error:", e)
