import json

with open("modern_gpt/diagnostics/phase0/training_audit_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== STEP 8D LOG SUMMARY ===")
s8d = data.get("step_08d_log_summary", {})
for k, v in s8d.items():
    if k != "epochs":
        print(f"  {k}: {v}")
print("\nEpochs detail:")
for ep, stats in s8d.get("epochs", {}).items():
    print(f"  {ep}: {stats}")

print("\n=== STEP 7B LOG SUMMARY ===")
s7b = data.get("step_07b_log_summary", {})
for k, v in s7b.items():
    if k != "epochs":
        print(f"  {k}: {v}")

print("\n=== MASKING VERIFICATION ===")
for m in data.get("masking_verification", []):
    print(m)

