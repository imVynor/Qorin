"""Batch evaluation script across extended convergence checkpoints."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, ".")
from scripts.evaluate_midscale_run import evaluate_checkpoint

def run_trajectory_evaluation():
    checkpoints = [
        ("step_200", Path("checkpoints/sft_midscale_500_extended/step_200.pt")),
        ("best_val_step_300", Path("checkpoints/sft_midscale_500_extended/best_val.pt")),
        ("step_400", Path("checkpoints/sft_midscale_500_extended/step_400.pt")),
        ("final_step_600", Path("checkpoints/sft_midscale_500_extended/latest.pt")),
    ]

    out_dir = Path("artifacts/diagnostics/extended_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for name, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_path} (not found)")
            continue
        report_path = out_dir / f"{name}_eval.json"
        print(f"\n==================================================")
        print(f"EVALUATING: {name} ({ckpt_path})")
        print(f"==================================================")
        evaluate_checkpoint(ckpt_path, report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_records.append({
            "name": name,
            "checkpoint": str(ckpt_path),
            "train_memorization": data["memorization_train_summary"],
            "val_generalization": data["generalization_val_summary"],
        })

    summary_file = out_dir / "extended_trajectory_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n==================================================")
    print("CONSOLIDATED TRAJECTORY SUMMARY:")
    print("==================================================")
    print(f"{'Checkpoint':<18} | {'Py AST (Tr/Val)':<18} | {'Py Func (Tr/Val)':<18} | {'JSON Syn (Tr/Val)':<18} | {'JSON Sch (Tr/Val)':<18} | {'QA (Tr/Val)':<12}")
    print("-" * 115)
    for s in summary_records:
        tr = s["train_memorization"]
        vl = s["val_generalization"]
        py_ast = f"{tr['python']['ast_valid'].split()[0]} / {vl['python']['ast_valid'].split()[0]}"
        py_func = f"{tr['python']['functional_passed'].split()[0]} / {vl['python']['functional_passed'].split()[0]}"
        js_syn = f"{tr['json']['syntax_valid'].split()[0]} / {vl['json']['syntax_valid'].split()[0]}"
        js_sch = f"{tr['json']['schema_valid'].split()[0]} / {vl['json']['schema_valid'].split()[0]}"
        qa = f"{tr['qa']['accurate'].split()[0]} / {vl['qa']['accurate'].split()[0]}"
        print(f"{s['name']:<18} | {py_ast:<18} | {py_func:<18} | {js_syn:<18} | {js_sch:<18} | {qa:<12}")

if __name__ == "__main__":
    run_trajectory_evaluation()

