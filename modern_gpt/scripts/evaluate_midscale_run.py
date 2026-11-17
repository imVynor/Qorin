"""Evaluation suite for ModernGPT-S Midscale 500 Experiment.

Evaluates:
- Memorization (Train split: 400 examples)
- Generalization (Val split: 100 held-out examples)
Across:
- JSON syntax and schema validity
- Python AST syntax and sandbox functional test pass rate
- QA accuracy
- Explanation coherence and token diversity
"""

import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, ".")
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.evaluate_sft import generate_sft_completion, extract_code_block, extract_json_block, compute_diversity
from scripts.generate_midscale_dataset import test_code_snippet

DATA_DIR = Path("data/midscale_500")

def evaluate_checkpoint(ckpt_path: Path, output_json_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")

    print(f"Loading checkpoint: {ckpt_path} on {device}...")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(
        vocab_size=state["vocab_size"],
        context_length=state["model_config"].get("context_length", 512),
        emb_dim=state["model_config"].get("emb_dim", 384),
        n_heads=state["model_config"].get("n_heads", 6),
        n_layers=state["model_config"].get("n_layers", 6),
        ffn_hidden_dim=state["model_config"].get("ffn_hidden_dim", 1024),
        dropout=state["model_config"].get("dropout", 0.1),
        qkv_bias=state["model_config"].get("qkv_bias", False),
        rope_theta=state["model_config"].get("rope_theta", 10000.0),
    )
    model = ModernGPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    with open(DATA_DIR / "unit_tests.json", "r", encoding="utf-8") as f:
        test_bundle = json.load(f)
    py_tests = test_bundle["python_tests"]
    json_schemas = test_bundle["json_schemas"]

    def eval_records(split_name: str, file_path: Path):
        print(f"\nEvaluating {split_name} from {file_path}...")
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        results = []
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

            py_v = None
            py_p = None
            js_v = None
            js_s = None
            qa_match = None

            if cat == "python":
                py_total += 1
                code = extract_code_block(gen_text)
                try:
                    ast.parse(code)
                    py_v = True
                    py_ast_valid += 1
                    spec = py_tests.get(rid, {})
                    tests = spec.get("tests", [])
                    test_code_snippet(code, tests)
                    py_p = True
                    py_func_pass += 1
                except Exception:
                    if py_v is None:
                        py_v = False
                    py_p = False

            elif cat == "json":
                json_total += 1
                js = extract_json_block(gen_text)
                try:
                    parsed = json.loads(js)
                    js_v = True
                    json_syntax_valid += 1
                    req_keys = json_schemas.get(rid, {}).get("required_keys", [])
                    if isinstance(parsed, dict) and all(k in parsed for k in req_keys):
                        js_s = True
                        json_schema_valid += 1
                    else:
                        js_s = False
                except Exception:
                    js_v = False
                    js_s = False

            elif cat == "qa":
                qa_total += 1
                # Check exact or normalized containment
                norm_gen = gen_text.lower().strip()
                norm_tgt = target.lower().strip()
                qa_match = (norm_tgt in norm_gen) or (norm_gen in norm_tgt)
                if qa_match:
                    qa_exact += 1

            elif cat == "explanation":
                exp_total += 1
                div = compute_diversity(gen_text)
                exp_diversities.append(div)

            results.append({
                "id": rid,
                "category": cat,
                "task_family": r["task_family"],
                "instruction": inst,
                "input": inp,
                "target": target,
                "generated": gen_text,
                "tokens_generated": meta["tokens_generated"],
                "stopped_on_eos": meta["stopped_on_eos"],
                "python_ast_valid": py_v,
                "python_functional_passed": py_p,
                "json_syntax_valid": js_v,
                "json_schema_valid": js_s,
                "qa_accurate": qa_match,
            })

            if (idx + 1) % 50 == 0 or idx == len(records) - 1:
                print(f"  Processed {idx + 1}/{len(records)} examples...")

        # Compute per-task-family breakdown
        by_family = {}
        for r in results:
            fam = r["task_family"]
            cat = r["category"]
            if fam not in by_family:
                by_family[fam] = {
                    "category": cat,
                    "total": 0,
                    "syntax_valid": 0,
                    "passed": 0,
                }
            by_family[fam]["total"] += 1
            if cat == "python":
                if r["python_ast_valid"]:
                    by_family[fam]["syntax_valid"] += 1
                if r["python_functional_passed"]:
                    by_family[fam]["passed"] += 1
            elif cat == "json":
                if r["json_syntax_valid"]:
                    by_family[fam]["syntax_valid"] += 1
                if r["json_schema_valid"]:
                    by_family[fam]["passed"] += 1
            elif cat == "qa":
                by_family[fam]["syntax_valid"] += 1
                if r["qa_accurate"]:
                    by_family[fam]["passed"] += 1
            elif cat == "explanation":
                by_family[fam]["syntax_valid"] += 1
                if r["stopped_on_eos"]:
                    by_family[fam]["passed"] += 1

        for fam, d in by_family.items():
            d["pass_rate"] = f"{d['passed']}/{d['total']} ({d['passed']/d['total']*100:.1f}%)" if d['total'] else "0.0%"

        summary = {
            "total_examples": len(records),
            "python": {
                "total": py_total,
                "ast_valid": f"{py_ast_valid}/{py_total} ({py_ast_valid/py_total*100:.1f}%)" if py_total else "N/A",
                "functional_passed": f"{py_func_pass}/{py_total} ({py_func_pass/py_total*100:.1f}%)" if py_total else "N/A",
            },
            "json": {
                "total": json_total,
                "syntax_valid": f"{json_syntax_valid}/{json_total} ({json_syntax_valid/json_total*100:.1f}%)" if json_total else "N/A",
                "schema_valid": f"{json_schema_valid}/{json_total} ({json_schema_valid/json_total*100:.1f}%)" if json_total else "N/A",
            },
            "qa": {
                "total": qa_total,
                "accurate": f"{qa_exact}/{qa_total} ({qa_exact/qa_total*100:.1f}%)" if qa_total else "N/A",
            },
            "explanation": {
                "total": exp_total,
                "avg_distinct_1": round(sum(d["distinct_1"] for d in exp_diversities) / len(exp_diversities), 4) if exp_diversities else 0.0,
                "avg_distinct_2": round(sum(d["distinct_2"] for d in exp_diversities) / len(exp_diversities), 4) if exp_diversities else 0.0,
            },
            "by_task_family": by_family,
        }
        return summary, results

    train_summary, train_details = eval_records("Train (Memorization)", DATA_DIR / "train.jsonl")
    val_summary, val_details = eval_records("Validation (Generalization)", DATA_DIR / "val.jsonl")

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "ModernGPT-S Midscale 500 Capability Run",
        "checkpoint": str(ckpt_path),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "memorization_train_summary": train_summary,
        "generalization_val_summary": val_summary,
        "train_details": train_details,
        "val_details": val_details,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY:")
    print("MEMORIZATION (TRAIN 400):")
    print(json.dumps(train_summary, indent=2))
    print("\nGENERALIZATION (VAL 100):")
    print(json.dumps(val_summary, indent=2))
    print(f"\nSaved full evaluation report to: {output_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/sft_midscale_500/best_val.pt")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/diagnostics/midscale_500_evaluation.json")
    evaluate_checkpoint(ckpt, out)
