import json
import urllib.request
import urllib.error
import time
from pathlib import Path

API_URL = "http://127.0.0.1:8000/api/chat/stream"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def send_chat_stream(payload):
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    
    events = []
    with urllib.request.urlopen(req, timeout=40.0) as resp:
        buffer = ""
        current_event = None
        for line_bytes in resp:
            line = line_bytes.decode("utf-8")
            if line.startswith("event:"):
                current_event = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str = line.replace("data:", "").strip()
                try:
                    data_obj = json.loads(data_str)
                except Exception:
                    data_obj = data_str
                events.append({"event": current_event, "data": data_obj})
            elif line.strip() == "":
                current_event = None
    return events

def run_live_demonstrations():
    print("=================================================================")
    print("QORIN Phase 3 — Live Application Demonstrations")
    print("Testing Live SSE Stream against http://127.0.0.1:8000/api/chat/stream")
    print("=================================================================")

    flows = [
        {
            "flow_num": 1,
            "title": "Standalone QORIN Conversation (Both Tools Disabled)",
            "model_id": "qorin-500m-sft-exp-c",
            "prompt": "Hello QORIN! Can you explain what a transformer attention mechanism is in two sentences?",
            "web_research_enabled": False,
            "python_sandbox_enabled": False,
            "expected_behavior": "Strict direct parametric response without any tool calls."
        },
        {
            "flow_num": 2,
            "title": "Python Execution with Only Python Enabled",
            "model_id": "qorin-500m-sft-exp-c",
            "prompt": "Compute 4321 * 8765 using python.",
            "web_research_enabled": False,
            "python_sandbox_enabled": True,
            "expected_behavior": "Model emits python call, executes in sandbox, outputs 37873565."
        },
        {
            "flow_num": 3,
            "title": "Web Retrieval with Only Web Enabled",
            "model_id": "qorin-500m-sft-exp-c",
            "prompt": "Search for the latest news on NASA Artemis program.",
            "web_research_enabled": True,
            "python_sandbox_enabled": False,
            "expected_behavior": "Model emits web_research call, returns sources or handles web fallback gracefully."
        },
        {
            "flow_num": 4,
            "title": "Tool Execution Followed by Second-Stage Model Synthesis",
            "model_id": "qorin-500m-sft-exp-c",
            "prompt": "Calculate the greatest common divisor of 123456 and 7890 using python math.gcd.",
            "web_research_enabled": False,
            "python_sandbox_enabled": True,
            "expected_behavior": "Model executes gcd, receives tool observation 6, synthesizes final answer stating 6."
        },
        {
            "flow_num": 5,
            "title": "Failed Tool Execution & Honest Error Recovery",
            "model_id": "qorin-500m-sft-exp-c",
            "prompt": "Execute python with division by zero: print(1 / 0)",
            "web_research_enabled": False,
            "python_sandbox_enabled": True,
            "expected_behavior": "Sandbox catches ZeroDivisionError, formats tool observation with error, model recovers."
        },
        {
            "flow_num": 6,
            "title": "Negative-Control Query with Both Tools Active",
            "model_id": "qorin-500m-sft-exp-c",
            "prompt": "Write a short haiku about spring blossoms falling into a river.",
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "expected_behavior": "Model generates creative direct response without triggering tools."
        }
    ]

    records = []

    for f in flows:
        fnum = f["flow_num"]
        print(f"\n--- Executing Flow {fnum}: {f['title']} ---")
        
        payload = {
            "model_id": f["model_id"],
            "messages": [{"role": "user", "content": f["prompt"]}],
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_new_tokens": 256,
            "repetition_penalty": 1.15,
            "retrieval_mode": "off",
            "retrieval_top_k": 3,
            "retrieval_threshold": 0.15,
            "web_research_enabled": f["web_research_enabled"],
            "python_sandbox_enabled": f["python_sandbox_enabled"]
        }

        start_t = time.perf_counter()
        try:
            events = send_chat_stream(payload)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            # Parse stream events
            tokens = []
            tool_results = []
            done_info = {}
            for ev in events:
                ename = ev.get("event")
                edata = ev.get("data")
                if ename == "token" and isinstance(edata, dict):
                    tokens.append(edata.get("text", ""))
                elif ename == "tool_result" and isinstance(edata, dict):
                    tool_results.append(edata)
                elif ename == "done" and isinstance(edata, dict):
                    done_info = edata

            final_text = "".join(tokens)

            rec = {
                "flow_num": fnum,
                "title": f["title"],
                "model_id": f["model_id"],
                "prompt": f["prompt"],
                "capabilities": {
                    "web_research_enabled": f["web_research_enabled"],
                    "python_sandbox_enabled": f["python_sandbox_enabled"]
                },
                "total_latency_ms": round(elapsed_ms, 2),
                "tool_executions_count": len(tool_results),
                "tool_traces": tool_results,
                "final_synthesized_response": final_text.strip(),
                "telemetry": done_info,
                "status": "SUCCESS"
            }
            print(f"Flow {fnum} Completed in {elapsed_ms:.1f}ms. Tools executed: {len(tool_results)}")
            print(f"Response: {final_text.strip()[:160]}...")
            records.append(rec)

        except Exception as err:
            print(f"Flow {fnum} FAILED: {err}")
            rec = {
                "flow_num": fnum,
                "title": f["title"],
                "prompt": f["prompt"],
                "error": str(err),
                "status": "ERROR"
            }
            records.append(rec)

    out_file = REPO_ROOT / "modern_gpt/reports/application_reliability_audit/live_demonstration_records.json"
    with open(out_file, "w", encoding="utf-8") as fl:
        json.dump({"live_demonstrations": records}, fl, indent=2)

    print(f"\nAll live demonstrations recorded in: {out_file}")

if __name__ == "__main__":
    run_live_demonstrations()
