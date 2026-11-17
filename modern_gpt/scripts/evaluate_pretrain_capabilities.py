"""Run held-out capability evaluation on pre-trained 16M and 50M checkpoints.

Evaluates 100 held-out examples from data/midscale_500/val.jsonl:
- Python AST syntax validity + sandbox functional test pass rate
- JSON syntax validity + JSON schema required keys compliance
- QA exact match / containment
- Explanation token diversity (Distinct-1, Distinct-2)
"""

import ast
import json
from pathlib import Path
import sys
import time
import torch

# Add repository root to sys.path
sys.path.insert(0, ".")

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.evaluate_sft import generate_sft_completion, extract_code_block, extract_json_block, compute_diversity
from scripts.generate_midscale_dataset import test_code_snippet

DATA_DIR = Path("data/midscale_500")

def evaluate_checkpoint_capabilities(ckpt_path: Path, arch: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")

    print(f"\n--- Evaluating {arch.upper()} from {ckpt_path} ---")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("model_config", {})
    config = ModelConfig(
        vocab_size=state.get("vocab_size", 16000),
        context_length=cfg_dict.get("context_length", 512),
        emb_dim=cfg_dict.get("emb_dim", 384 if arch == "16m" else 512),
        n_heads=cfg_dict.get("n_heads", 6 if arch == "16m" else 8),
        n_layers=cfg_dict.get("n_layers", 6 if arch == "16m" else 12),
        ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 1024 if arch == "16m" else 1536),
        dropout=cfg_dict.get("dropout", 0.1),
        qkv_bias=cfg_dict.get("qkv_bias", False),
        rope_theta=cfg_dict.get("rope_theta", 10000.0),
    )
    model = ModernGPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    with open(DATA_DIR / "unit_tests.json", "r", encoding="utf-8") as f:
        test_bundle = json.load(f)
    py_tests = test_bundle["python_tests"]
    json_schemas = test_bundle["json_schemas"]

    records = []
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    py_total = 0
    py_ast_valid = 0
    py_func_pass = 0

    json_total = 0
    json_syntax_valid = 0
    json_schema_valid = 0

    qa_total = 0
    qa_exact = 0

    exp_total = 0
    exp_diversities = []

    for idx, r in enumerate(records):
        cat = r["category"]
        rid = r["id"]
        inst = r["instruction"]
        inp = r.get("input", "")
        target = r["output"].strip()

        resp, meta = generate_sft_completion(
            model, tok, inst, inp, max_new_tokens=128, temperature=0.0, device=device
        )
        gen_text = resp.strip()

        if cat == "python":
            py_total += 1
            code = extract_code_block(gen_text)
            try:
                ast.parse(code)
                py_ast_valid += 1
                spec = py_tests.get(rid, {})
                tests = spec.get("tests", [])
                test_code_snippet(code, tests)
                py_func_pass += 1
            except Exception:
                pass

        elif cat == "json":
            json_total += 1
            js = extract_json_block(gen_text)
            try:
                parsed = json.loads(js)
                json_syntax_valid += 1
                req_keys = json_schemas.get(rid, {}).get("required_keys", [])
                if isinstance(parsed, dict) and all(k in parsed for k in req_keys):
                    json_schema_valid += 1
            except Exception:
                pass

        elif cat == "qa":
            qa_total += 1
            norm_gen = gen_text.lower().strip()
            norm_tgt = target.lower().strip()
            if (norm_tgt in norm_gen) or (norm_gen in norm_tgt):
                qa_exact += 1

        elif cat == "explanation":
            exp_total += 1
            div = compute_diversity(gen_text)
            exp_diversities.append(div)

        if (idx + 1) % 25 == 0 or idx == len(records) - 1:
            print(f"  Processed {idx + 1}/{len(records)} examples...")

    summary = {
        "checkpoint": str(ckpt_path),
        "architecture": arch,
        "total_held_out": len(records),
        "python": {
            "total": py_total,
            "ast_valid": f"{py_ast_valid}/{py_total} ({py_ast_valid/py_total*100:.1f}%)",
            "functional_passed": f"{py_func_pass}/{py_total} ({py_func_pass/py_total*100:.1f}%)",
        },
        "json": {
            "total": json_total,
            "syntax_valid": f"{json_syntax_valid}/{json_total} ({json_syntax_valid/json_total*100:.1f}%)",
            "schema_valid": f"{json_schema_valid}/{json_total} ({json_schema_valid/json_total*100:.1f}%)",
        },
        "qa": {
            "total": qa_total,
            "accurate": f"{qa_exact}/{qa_total} ({qa_exact/qa_total*100:.1f}%)",
        },
        "explanation": {
            "total": exp_total,
            "avg_distinct_1": round(sum(d["distinct_1"] for d in exp_diversities) / len(exp_diversities), 4) if exp_diversities else 0.0,
            "avg_distinct_2": round(sum(d["distinct_2"] for d in exp_diversities) / len(exp_diversities), 4) if exp_diversities else 0.0,
        },
    }
    return summary


def main():
    ckpts = [
        ("checkpoints/exp_100m/best_val.pt", "16m"),
        ("checkpoints/pretrain_50m_250m/best_val.pt", "50m"),
    ]

    all_results = {}
    for path, arch in ckpts:
        p = Path(path)
        if p.exists():
            res = evaluate_checkpoint_capabilities(p, arch)
            all_results[arch] = res
            print(json.dumps(res, indent=2))
        else:
            print(f"Warning: {path} not found!")

    out_file = Path("artifacts/reports/capability_16m_vs_50m_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved capability comparison to {out_file}")

if __name__ == "__main__":
    main()

