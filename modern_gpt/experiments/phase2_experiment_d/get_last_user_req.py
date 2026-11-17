import json

transcript_path = r'C:\Users\Anish\.gemini\antigravity\brain\cca78c81-05e3-4cea-bf48-9597dd31c3e4\.system_generated\logs\transcript_full.jsonl'
last_user_input = None
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get('type') == 'USER_INPUT':
                last_user_input = obj.get('content')
        except Exception:
            pass

with open('modern_gpt/experiments/phase2_experiment_d/last_user_request.md', 'w', encoding='utf-8') as out:
    out.write(last_user_input if last_user_input else 'None found')
print("Wrote prompt to modern_gpt/experiments/phase2_experiment_d/last_user_request.md")
