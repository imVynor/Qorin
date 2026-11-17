import requests
import json
from typing import Tuple, Optional

def test_trained_tool_selection(user_q, web_avail, py_avail):
    payload = {
        "model_id": "qorin-500m-sft-exp-c",
        "messages": [{"role": "user", "content": user_q}],
        "temperature": 0.0,
        "max_new_tokens": 120,
        "python_sandbox_enabled": py_avail,
        "web_research_enabled": web_avail
    }
    r = requests.post("http://127.0.0.1:8000/api/chat/stream", json=payload, stream=True)
    
    first_round_text = ""
    tool_starts = []
    tool_results = []
    tokens = []
    
    for line in r.iter_lines():
        if line and line.decode().startswith("data:"):
            data_str = line.decode()[5:].strip()
            try:
                obj = json.loads(data_str)
                if isinstance(obj, dict):
                    if "tool_name" in obj and "status" in obj and obj["status"] == "started":
                        tool_starts.append(obj)
                    elif "tool_name" in obj and "output_payload" in obj:
                        tool_results.append(obj)
                    elif "text" in obj:
                        tokens.append(obj["text"])
            except Exception:
                pass
    return tool_starts, tool_results, "".join(tokens)

queries = [
    ("What is the capital of France?", True, True),
    ("Calculate (144 * 12) + (864 / 16) using Python.", True, True),
    ("Search the web for the authors of the Attention Is All You Need paper.", True, True),
    ("Search the web for the authors of the Attention Is All You Need paper.", True, False), # Web only
    ("Write a haiku about autumn leaves falling.", True, True),
    ("Compute the mean and standard deviation of [12, 18, 24, 30] in Python.", True, True),
    ("Calculate 25 * 25 using Python.", False, True), # Python only
    ("Calculate 25 * 25 using Python.", False, False) # Standalone (both False)
]

for q, w, p in queries:
    starts, results, text = test_trained_tool_selection(q, w, p)
    tool_called = starts[0]["tool_name"] if starts else "NONE (DIRECT)"
    print(f"Q: {q[:60]}")
    print(f"   [Config: Web={w}, Py={p}] -> Action: {tool_called}")
    if results:
        print(f"   Execution status: {results[0].get('status')}")
    print(f"   Final snippet: {repr(text[:90])}\n")
