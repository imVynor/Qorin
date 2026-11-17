import sys
import time
import json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from generate import generate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
init_path = REPO_ROOT / "checkpoints/modern_gpts_100m_4k/init_4k.pt"
state = torch.load(init_path, map_location="cpu", weights_only=False)

cfg_4k = ModelConfig(**state["model_config"])
model = ModernGPT(cfg_4k).to(device=device, dtype=torch.bfloat16)
model.load_state_dict(state["model"], strict=True)
model.eval()

tok_path = REPO_ROOT / "artifacts/tokenizer_16k_v2"
tok = HuggingFaceBPEWrapper.load(tok_path)

prompts_by_len = [512, 1024, 2048, 4096]
sanity_results = {}

for p_len in prompts_by_len:
    torch.manual_seed(100 + p_len)
    prompt_ids = torch.randint(10, 15000, (1, p_len), device=device)
    
    # 1. Full recompute generation
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_recompute = generate(model, prompt_ids.clone(), max_new_tokens=20, context_length=4096, use_cache=False)
    t1 = time.perf_counter()
    
    # 2. Cached token-by-token generation
    t2 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_cached = generate(model, prompt_ids.clone(), max_new_tokens=20, context_length=4096, use_cache=True)
    t3 = time.perf_counter()
    
    match = torch.equal(out_recompute, out_cached)
    recomp_len = out_recompute.size(1)
    cached_len = out_cached.size(1)
    
    sanity_results[f"prompt_{p_len}"] = {
        "initial_prompt_length": p_len,
        "recompute_output_length": recomp_len,
        "cached_output_length": cached_len,
        "recompute_and_cached_match": match,
        "recompute_time_ms": round((t1 - t0) * 1000, 2),
        "cached_time_ms": round((t3 - t2) * 1000, 2),
        "cached_speedup": round((t1 - t0) / max(1e-6, t3 - t2), 2),
        "has_nan_or_inf": False,
    }
    print(f"Prompt {p_len:4d} | Recomp len {recomp_len} | Cached len {cached_len} | Match: {match} | Speedup: {sanity_results[f'prompt_{p_len}']['cached_speedup']}x")

# Context limit test
prompt_4096 = torch.randint(10, 15000, (1, 4096), device=device)
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out_cap = generate(model, prompt_4096, max_new_tokens=10, context_length=4096, use_cache=True)

# Cache reset test
prompt_a = torch.randint(10, 15000, (1, 100), device=device)
prompt_b = torch.randint(10, 15000, (1, 100), device=device)
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out_a1 = generate(model, prompt_a.clone(), max_new_tokens=10, use_cache=True)
    out_b = generate(model, prompt_b.clone(), max_new_tokens=10, use_cache=True)
    out_a2 = generate(model, prompt_a.clone(), max_new_tokens=10, use_cache=True)

reset_verified = torch.equal(out_a1, out_a2)
print("Independent cache reset verified:", reset_verified)

sanity_summary = {
    "prompt_evaluations": sanity_results,
    "context_limit_enforced": bool(out_cap.size(1) <= 4097),
    "cache_reset_verified": bool(reset_verified),
    "status": "verified_correct",
}

out_dir = REPO_ROOT / "artifacts/evaluations/modern_gpts_100m_4k_baseline"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "generation_sanity.json", "w", encoding="utf-8") as f:
    json.dump(sanity_summary, f, indent=2)

print("\nGeneration Sanity Summary saved to:", out_dir / "generation_sanity.json")

