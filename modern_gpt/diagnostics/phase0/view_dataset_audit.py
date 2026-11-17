import json

with open("modern_gpt/diagnostics/phase0/dataset_audit_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for d_name, d_res in data.items():
    print(f"\n==================== {d_name} ====================")
    print(f"Total records: {d_res['total_records']}")
    print(f"Single turn: {d_res['single_turn_records']} ({d_res['single_turn_pct']}%)")
    print(f"Multiturn: {d_res['multiturn_records']} ({d_res['multiturn_pct']}%)")
    print("Turn distribution:", d_res["turn_distribution"])
    print("Categories:", d_res["categories"])
    print("Sources:", d_res["sources"])
    print("Prompt tokens stats:", d_res["prompt_tokens_stats"])
    print("Response tokens stats:", d_res["response_tokens_stats"])
    print("Response length buckets:", d_res["response_length_buckets"])
    print("Top openings:", d_res["top_openings"])
    print("Attractor matches:", d_res["attractor_matches"])
    print("Greetings found count:", d_res["greetings_found_count"])
    if d_res["greeting_samples"]:
        print("\n--- Greeting Samples ---")
        for g in d_res["greeting_samples"][:4]:
            print(f"User: {repr(g['user'])}")
            print(f"Assistant: {repr(g['assistant'][:200])}")
            print(f"Source: {g.get('source')}, Cat: {g.get('category')}")
    if d_res["email_samples"]:
        print("\n--- Email Boilerplate Samples ---")
        for e in d_res["email_samples"][:3]:
            print(f"User: {repr(e['user'])}")
            print(f"Assistant: {repr(e['assistant'][:200])}")

