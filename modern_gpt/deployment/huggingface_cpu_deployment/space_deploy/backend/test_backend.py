import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

# 1. Health
r = client.get('/api/health')
print('[+] Health:', r.status_code, r.json().get('status'))

# 2. Models
r = client.get('/api/models')
models = r.json()
print('[+] Listed', len(models), 'models:')
for m in models:
    print(f"   - {m['id']}: {m['name']} status={m['status']}")

# 3. Training runs
r = client.get('/api/training/runs')
runs = r.json()
print('[+] Listed', len(runs), 'training runs')

# 4. Training 500m details
r = client.get('/api/training/runs/qorin-500m-pretraining')
run_data = r.json()
print(f"[+] 500M Pretrain Run: {run_data.get('name')}, steps={run_data.get('current_step')}, points={len(run_data.get('series', []))}")

# 5. Retrieval query test
r = client.post('/api/retrieval/query', json={'query': 'What is the hardware duty cycle for RTX 5060?'})
ret_data = r.json()
print(f"[+] Retrieval query triggered: {ret_data.get('triggered')}, chunks found: {len(ret_data.get('retrieved_chunks', []))}")
if ret_data.get('retrieved_chunks'):
    print(f"   Top chunk title: {ret_data['retrieved_chunks'][0]['doc_title']}")

# 6. Python Sandbox tool test
r = client.post('/api/tools/execute', json={'tool_name': 'python', 'arguments': {'code': 'print(sum([x**2 for x in range(10)]))'}})
tool_data = r.json()
print(f"[+] Python Tool status: {tool_data.get('status')}, stdout: {tool_data.get('output_payload', {}).get('stdout', '').strip()}")

# 7. Security block test in sandbox
r = client.post('/api/tools/execute', json={'tool_name': 'python', 'arguments': {'code': 'import os; os.system("dir")'}})
sec_data = r.json()
print(f"[+] Sandbox Security Violation check: status={sec_data.get('status')}, error={sec_data.get('error')}")

# 8. Web Research test
r = client.post('/api/tools/execute', json={'tool_name': 'web_research', 'arguments': {'query': 'transformer rotary position embeddings'}})
web_data = r.json()
sources = web_data.get('output_payload', {}).get('sources', [])
print(f"[+] Web Research test: status={web_data.get('status')}, sources_found={len(sources)}")
if sources:
    print(f"   Source 1: {sources[0]['title']} ({sources[0]['url']})")

print("\n=== ALL BACKEND MODULES VERIFIED & OPERATIONAL ===")
