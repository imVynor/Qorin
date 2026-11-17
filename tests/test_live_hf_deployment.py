"""
QORIN Live Hugging Face CPU Space Integration & End-to-End Test Suite
Tests live remote endpoints on https://vynor-stt4.hf.space
"""

import json
import os
import sys
import time
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_URL = "https://vynor-stt4.hf.space"

# Load local secret for testing (never printed)
sec_path = REPO_ROOT / "backend/.secrets.json"
with open(sec_path, "r", encoding="utf-8") as f:
    sec = json.load(f)
EVAL_KEY = sec["COMPUTE_ACCESS_PASSWORD"]


def test_1_health_and_telemetry():
    print("[Live 1/9] Testing Live Health & Telemetry Endpoint...")
    res = requests.get(f"{BASE_URL}/api/health", timeout=15)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["status"] == "healthy"
    telemetry = data["telemetry"]
    assert telemetry["qorin"]["device"] == "cpu"
    assert telemetry["qorin"]["cuda_available"] is False
    print("           PASS: Healthcheck 200 OK. Confirmed running on CPU Basic.")
    return telemetry


def test_2_model_list():
    print("[Live 2/9] Testing Live Model Catalog...")
    res = requests.get(f"{BASE_URL}/api/models", timeout=15)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    models = res.json()
    ids = [m["id"] for m in models]
    assert "qorin-100m-sft-v3" in ids
    assert "qorin-500m-sft-exp-f" in ids
    print(f"           PASS: Catalog returned {len(models)} models including 100M and 500M scales.")


def test_3_unauthorized_rejection():
    print("[Live 3/9] Testing Live Unauthorized Access Rejection...")
    res = requests.post(f"{BASE_URL}/api/models/select", json={"model_id": "qorin-100m-sft-v3"}, timeout=15)
    assert res.status_code == 401, f"Expected 401, got {res.status_code}"
    assert "Evaluator compute access protected" in res.json()["detail"]
    print("           PASS: Live Space rejected unauthenticated request with 401.")


def test_4_live_auth_and_token_issuance():
    print("[Live 4/9] Testing Live Evaluator Authentication...")
    res = requests.post(f"{BASE_URL}/api/auth/verify-compute", json={"password": EVAL_KEY}, timeout=15)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["success"] is True
    token = data["token"]
    assert token.startswith("qorin_tok_")
    print("           PASS: Live Space issued valid HMAC session token (zero secret disclosure).")
    return token


def test_5_hydrate_and_select_100m(session_token):
    print("[Live 5/9] Testing Live 100M Model Hydration from vynor/qorin-models & CPU Load...")
    headers = {"X-Compute-Token": session_token}
    t0 = time.time()
    res = requests.post(
        f"{BASE_URL}/api/models/select",
        json={"model_id": "qorin-100m-sft-v3", "device": "cpu"},
        headers=headers,
        timeout=180,
    )
    elapsed = time.time() - t0
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["success"] is True
    assert data["active_model_id"] == "qorin-100m-sft-v3"
    print(f"           PASS: 100M checkpoint hydrated and loaded into CPU RAM in {elapsed:.2f}s.")


def test_6_live_sse_inference_100m(session_token):
    print("[Live 6/9] Testing Live CPU Token Generation & SSE Streaming...")
    headers = {"X-Compute-Token": session_token, "Content-Type": "application/json"}
    payload = {
        "model_id": "qorin-100m-sft-v3",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0.2,
        "max_new_tokens": 20,
    }
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/api/chat/stream", json=payload, headers=headers, stream=True, timeout=60)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    
    received_tokens = []
    current_event = None
    for line in res.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("event: "):
                current_event = decoded[len("event: "):].strip()
            elif decoded.startswith("data: "):
                raw_json = decoded[len("data: "):]
                try:
                    chunk = json.loads(raw_json)
                    if current_event == "token" and "text" in chunk:
                        received_tokens.append(chunk["text"])
                    elif current_event == "done":
                        print(f"           Stream done event received: {chunk.get('total_tokens')} tokens in {chunk.get('total_latency_ms', chunk.get('elapsed_ms'))}ms")
                except Exception:
                    pass

    elapsed = time.time() - t0
    full_output = "".join(received_tokens)
    print(f"           PASS: Live CPU inference produced {len(received_tokens)} tokens in {elapsed:.2f}s.")
    print(f"           Sample Output: {full_output[:70]!r}...")
    assert len(received_tokens) > 0, "No tokens received from SSE stream"


def test_7_live_tool_execution(session_token):
    print("[Live 7/9] Testing Live Tool Execution on Remote Space...")
    headers = {"X-Compute-Token": session_token}
    payload = {
        "tool_name": "weather",
        "arguments": {"location": "London"}
    }
    res = requests.post(f"{BASE_URL}/api/tools/execute", json=payload, headers=headers, timeout=20)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data.get("status") == "completed" or "output_payload" in data or "stdout" in data
    out_text = str(data.get("output_payload") or data.get("stdout") or data)
    print(f"           PASS: Remote live tool executed: {out_text[:60]}...")


def test_8_model_unload(session_token):
    print("[Live 8/9] Testing Live Model Memory Unload...")
    headers = {"X-Compute-Token": session_token}
    res = requests.post(f"{BASE_URL}/api/models/unload", headers=headers, timeout=15)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["success"] is True
    assert data["telemetry"]["active_model_id"] is None
    print("           PASS: Model unloaded cleanly from remote Space RAM.")


def test_9_session_revocation(session_token):
    print("[Live 9/9] Testing Live Session Revocation / Logout...")
    headers = {"X-Compute-Token": session_token}
    res_logout = requests.post(f"{BASE_URL}/api/auth/logout", headers=headers, timeout=15)
    assert res_logout.status_code == 200, f"Expected 200, got {res_logout.status_code}"
    
    # Try using the revoked token on a protected endpoint
    res_rejected = requests.post(f"{BASE_URL}/api/models/select", json={"model_id": "qorin-100m-sft-v3"}, headers=headers, timeout=15)
    assert res_rejected.status_code == 401, f"Expected 401 for revoked token, got {res_rejected.status_code}"
    print("           PASS: Session revoked on live Space; subsequent requests rejected.")


if __name__ == "__main__":
    print("=" * 70)
    print("QORIN LIVE HUGGING FACE SPACE CPU INTEGRATION TEST")
    print(f"Target: {BASE_URL}")
    print("=" * 70)
    test_1_health_and_telemetry()
    test_2_model_list()
    test_3_unauthorized_rejection()
    token = test_4_live_auth_and_token_issuance()
    test_5_hydrate_and_select_100m(token)
    test_6_live_sse_inference_100m(token)
    test_7_live_tool_execution(token)
    test_8_model_unload(token)
    test_9_session_revocation(token)
    print("=" * 70)
    print("ALL 9 LIVE SPACE INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)
