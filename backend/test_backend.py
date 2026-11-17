import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'healthy'

def test_models_list():
    r = client.get('/api/models')
    assert r.status_code == 200
    models = r.json()
    assert len(models) >= 5

def test_training_runs():
    r = client.get('/api/training/runs')
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) >= 1

def test_retrieval_query():
    r = client.post('/api/retrieval/query', json={'query': 'What is the hardware duty cycle for RTX 5060?'})
    assert r.status_code == 200

if __name__ == '__main__':
    test_health()
    test_models_list()
    test_training_runs()
    test_retrieval_query()
    print("\n=== ALL BACKEND MODULES VERIFIED & OPERATIONAL ===")
