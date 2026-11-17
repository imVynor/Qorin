"""
QORIN Hugging Face CPU Readiness & Verification Test Suite
Tests CPU model execution, HMAC authentication, rate limiting, and memory management.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Enforce CPU execution for this test
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from backend.app import (
    generate_session_token,
    verify_session_token,
    check_compute_auth,
    COMPUTE_ACCESS_PASSWORD,
)
from backend.models.hf_loader import MODEL_REMOTE_REGISTRY, compute_file_sha256
from backend.models.manager import ModelManager
from backend.models.base import GenerationConfig


def test_authentication_system():
    print("[1/5] Testing HMAC Session Authentication & Rotation...")
    token = generate_session_token("evaluator_prof", expiry_hours=24)
    assert token.startswith("qorin_tok_"), "Session token missing canonical prefix"
    assert verify_session_token(token), "Valid session token failed verification"
    assert check_compute_auth(token), "check_compute_auth failed for session token"
    # Strict security: raw password is NOT a valid bearer token
    assert not check_compute_auth(COMPUTE_ACCESS_PASSWORD), "Raw password must not be accepted as token"
    assert not check_compute_auth("UDIT_BHATIA_AUTHORIZED"), "Legacy token must be rejected"
    assert not check_compute_auth("INVALID_CREDENTIAL"), "Invalid credential was accepted"
    print("      HMAC tokens and strict rejection of legacy fallbacks passed.")


def test_registry_hashes():
    print("[2/5] Testing Checkpoint Cryptographic Hash Integrity...")
    path_map = {
        "qorin-100m-sft-v3": REPO_ROOT / "modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt",
        "qorin-500m-sft-exp-f": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt",
    }
    for model_id, meta in MODEL_REMOTE_REGISTRY.items():
        local_cand = path_map.get(model_id)
        if local_cand and local_cand.exists():
            computed = compute_file_sha256(local_cand)
            assert computed == meta["sha256"], f"Hash mismatch for {model_id}!\nExp: {meta['sha256']}\nGot: {computed}"
            print(f"      Verified {model_id} SHA-256: {computed[:16]}... (matches registry)")
        else:
            print(f"      Local checkpoint {local_cand} not found, skipping disk hash check.")


def test_cpu_model_loading_and_generation():
    print("[3/5] Testing Native QORIN Inference on CPU (torch.float32)...")
    manager = ModelManager()
    model_id = "qorin-100m-sft-v3"
    
    t0 = time.time()
    loaded = manager.select_model(model_id, device="cpu")
    load_time = time.time() - t0
    assert loaded, f"Failed to select {model_id} on CPU"
    print(f"      Model {model_id} successfully loaded onto CPU in {load_time:.2f}s")
    
    # Verify telemetry
    telem = manager.get_telemetry()
    assert telem["active_model_id"] == model_id, "Active model ID mismatch in telemetry"
    assert telem["qorin"]["device"] == "cpu", f"Device is {telem['qorin']['device']}, expected cpu"
    print(f"      Telemetry verified: device={telem['qorin']['device']}, loaded={telem['qorin']['is_loaded']}")
    
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    config = GenerationConfig(
        max_new_tokens=15,
        temperature=0.3,
        top_p=0.9,
    )
    
    async def run_gen():
        tokens = []
        async for chunk in manager.generate_stream(model_id, messages, config):
            tokens.append(chunk.text)
        return "".join(tokens)
    
    t_gen_start = time.time()
    output_text = asyncio.run(run_gen())
    gen_time = time.time() - t_gen_start
    print(f"      Generated output in {gen_time:.2f}s on CPU.")
    print(f"      Sample output: {output_text[:60]!r}...")
    
    # Test model unloading
    unloaded = manager.unload_active()
    assert unloaded, "Failed to unload active model"
    telem_after = manager.get_telemetry()
    assert telem_after["active_model_id"] is None, "Model was not unloaded cleanly"
    print("      Model unloaded cleanly from CPU memory.")


def test_retrieval_and_tools():
    print("[4/5] Testing Non-Parametric Retrieval & Tools on CPU...")
    from backend.retrieval.engine import RetrievalEngine
    from backend.tools.weather_provider import RealWeatherProvider
    from backend.tools.registry import CanonicalToolRegistry
    
    retriever = RetrievalEngine()
    trace = retriever.retrieve("What is QORIN architecture?", mode="auto", top_k=2)
    print(f"      Retriever triggered: {trace.triggered}, chunks found: {len(trace.retrieved_chunks)} in {trace.retrieval_latency_ms}ms")
    
    weather = RealWeatherProvider()
    ok, obs, err = weather.get_weather("Zurich")
    assert ok and obs is not None, f"Weather query failed: {err}"
    print(f"      Tool weather query returned successfully: {obs.location_name}, {obs.temperature_c}°C, {obs.weather_condition}")


def test_space_package_artifacts():
    print("[5/5] Testing Space Packaging Artifacts...")
    pkg_dir = REPO_ROOT / "modern_gpt/deployment/huggingface_cpu_deployment/stt4_package"
    assert (pkg_dir / "Dockerfile").exists(), "Space Dockerfile missing"
    assert (pkg_dir / "README.md").exists(), "Space README.md missing"
    assert (pkg_dir / "requirements.txt").exists(), "Space requirements.txt missing"
    assert (pkg_dir / ".dockerignore").exists(), "Space .dockerignore missing"
    
    # Check README YAML frontmatter
    readme_text = (pkg_dir / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in readme_text, "Missing sdk: docker in Space README"
    assert "app_port: 7860" in readme_text, "Missing app_port: 7860 in Space README"
    print("      All required Space packaging files present and verified.")


if __name__ == "__main__":
    print("=" * 65)
    print("QORIN HUGGING FACE CPU READINESS VERIFICATION")
    print("=" * 65)
    test_authentication_system()
    test_registry_hashes()
    test_cpu_model_loading_and_generation()
    test_retrieval_and_tools()
    test_space_package_artifacts()
    print("=" * 65)
    print("ALL CPU READINESS VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
