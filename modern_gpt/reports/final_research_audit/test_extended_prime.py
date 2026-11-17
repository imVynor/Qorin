import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8000/api/chat/stream"
MODEL_ID = "qorin-500m-sft-exp-c"

EXTENDED_PRIME_SUITE = [
    # Negatives (Expected False)
    {"val": -13, "expected": False, "category": "negative"},
    {"val": -7, "expected": False, "category": "negative"},
    {"val": -2, "expected": False, "category": "negative"},
    {"val": -1, "expected": False, "category": "negative"},
    
    # Boundary 0 and 1 (Expected False)
    {"val": 0, "expected": False, "category": "boundary"},
    {"val": 1, "expected": False, "category": "boundary"},
    
    # Small Primes (Expected True)
    {"val": 2, "expected": True, "category": "even_prime"},
    {"val": 3, "expected": True, "category": "small_prime"},
    {"val": 5, "expected": True, "category": "small_prime"},
    {"val": 7, "expected": True, "category": "small_prime"},
    {"val": 11, "expected": True, "category": "small_prime"},
    {"val": 13, "expected": True, "category": "small_prime"},
    {"val": 17, "expected": True, "category": "small_prime"},
    {"val": 19, "expected": True, "category": "small_prime"},
    {"val": 23, "expected": True, "category": "small_prime"},
    {"val": 29, "expected": True, "category": "small_prime"},
    {"val": 31, "expected": True, "category": "small_prime"},
    
    # Small Composites (Expected False)
    {"val": 4, "expected": False, "category": "small_composite"},
    {"val": 6, "expected": False, "category": "small_composite"},
    {"val": 8, "expected": False, "category": "small_composite"},
    {"val": 9, "expected": False, "category": "odd_composite"},
    {"val": 15, "expected": False, "category": "odd_composite"},
    {"val": 21, "expected": False, "category": "odd_composite"},
    {"val": 25, "expected": False, "category": "odd_square"},
    {"val": 27, "expected": False, "category": "odd_cube"},
    {"val": 35, "expected": False, "category": "semiprime"},
    {"val": 49, "expected": False, "category": "semiprime"},
    
    # Medium Primes & Composites
    {"val": 97, "expected": True, "category": "two_digit_prime"},
    {"val": 100, "expected": False, "category": "even_composite"},
    {"val": 101, "expected": True, "category": "three_digit_prime"},
    {"val": 119, "expected": False, "category": "semiprime_7x17"},
    {"val": 127, "expected": True, "category": "three_digit_prime"},
    {"val": 541, "expected": True, "category": "three_digit_prime"},
    {"val": 561, "expected": False, "category": "carmichael_number"},
    {"val": 1000, "expected": False, "category": "power_of_10"},
    {"val": 1009, "expected": True, "category": "four_digit_prime"},
    {"val": 1105, "expected": False, "category": "carmichael_number"}
]

def generate_code_for_is_prime():
    prompt = "Write a Python function `is_prime(n)` that returns True if n is a prime number, and False otherwise."
    payload = {
        "model_id": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "python_sandbox_enabled": False,
        "web_research_enabled": False,
        "temperature": 0.0,
        "max_new_tokens": 120
    }
    r = requests.post(API_URL, json=payload, stream=True, timeout=20)
    tokens = []
    for line in r.iter_lines():
        if line:
            l = line.decode("utf-8")
            if l.startswith("data:"):
                try:
                    obj = json.loads(l.replace("data:", "").strip())
                    if isinstance(obj, dict) and "text" in obj:
                        tokens.append(obj["text"])
                except Exception:
                    pass
    return "".join(tokens)

def test_code_against_suite(raw_code: str):
    print("Raw Generated Code from Exp C:")
    print(raw_code)
    print("=" * 50)
    
    code_to_exec = raw_code
    if "```python" in raw_code:
        code_to_exec = raw_code.split("```python")[1].split("```")[0]
    elif "```" in raw_code:
        code_to_exec = raw_code.split("```")[1].split("```")[0]

    tests_json = json.dumps(EXTENDED_PRIME_SUITE)
    harness = f"""
import json

{code_to_exec}

tests = json.loads('''{tests_json}''')
results = []
correct_count = 0

for t in tests:
    val = t['val']
    exp = t['expected']
    cat = t['category']
    try:
        actual = is_prime(val)
        ok = (actual == exp)
        if ok:
            correct_count += 1
        results.append({{"val": val, "expected": exp, "actual": actual, "correct": ok, "category": cat}})
    except Exception as e:
        results.append({{"val": val, "expected": exp, "error": str(e), "correct": False, "category": cat}})

print(json.dumps({{"total": len(tests), "correct": correct_count, "accuracy": round(correct_count/len(tests)*100, 2), "results": results}}))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(harness)
        tf_path = tf.name

    res = subprocess.run([sys.executable, tf_path], capture_output=True, text=True)
    try:
        out = json.loads(res.stdout)
        return out, code_to_exec
    except Exception as e:
        print("Harness failed to run:", res.stderr)
        return {"error": res.stderr, "raw_stdout": res.stdout}, code_to_exec

if __name__ == "__main__":
    code = generate_code_for_is_prime()
    eval_res, cleaned_code = test_code_against_suite(code)
    print("\nExtended is_prime Suite Results:")
    print(f"Total: {eval_res.get('total')}, Correct: {eval_res.get('correct')}, Accuracy: {eval_res.get('accuracy')}%")
    
    # Save results
    save_path = Path("modern_gpt/reports/final_research_audit/extended_prime_audit.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"generated_code": cleaned_code, "evaluation": eval_res}, f, indent=2)
    print(f"Saved {save_path}")
