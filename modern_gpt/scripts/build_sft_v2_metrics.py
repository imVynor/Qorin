import json
from pathlib import Path

# Load evaluation files
best_val_eval_path = Path("artifacts/diagnostics/sft_50m_targeted_v2_best_val_eval.json")
latest_eval_path = Path("artifacts/diagnostics/sft_50m_targeted_v2_latest_eval.json")

with open(best_val_eval_path, "r", encoding="utf-8") as f:
    best_val_data = json.load(f)

with open(latest_eval_path, "r", encoding="utf-8") as f:
    latest_data = json.load(f)

# Historical baselines
# 16M Base: Val loss 3.8615, Py AST 7.5%, Py Func 0.0%, JSON Syn 0.0%, QA 0.0%
# 16M SFT: Val loss 3.86, Py AST 77.5%, Py Func 0.0%, JSON Syn 67.5%, QA 0.0%
# 50M Base: Val loss 3.1152, Py AST 15.0%, Py Func 0.0%, JSON Syn 0.0%, QA 0.0%
# 50M SFT v1: Val loss 3.2615, Py AST 77.5%, Py Func 0.0%, JSON Syn 47.5%, QA 0.0%

metrics = {
    "experiment": "ModernGPT-S 50M SFT v2 Public Dataset Mixture",
    "architecture": "ModernGPT-S (49,105,408 parameters, 12 layers, 512 dim, 8 heads, 1536 ffn)",
    "base_checkpoint": "checkpoints/pretrain_50m_250m/latest.pt",
    "training_data": {
        "dataset": "data/sft_v2_public/train.jsonl",
        "total_records": 8597,
        "supervised_tokens": 1939120,
        "effective_batch_size": 8,
        "optimizer_steps": 3225,
        "epochs": 3.0,
        "learning_rate_schedule": "cosine_decay (2.5e-5 to 2.5e-6, 100 warmup steps)"
    },
    "training_dynamics": {
        "initial_val_loss": 1.5780,
        "initial_perplexity": 4.85,
        "best_val_step": 2400,
        "best_val_loss": 1.4713,
        "best_val_perplexity": 4.35,
        "final_val_loss": 1.4788,
        "final_perplexity": 4.39,
        "training_time_seconds": 573.51
    },
    "pretraining_loss_retention": {
        "evaluation_dataset": "artifacts/shards_250m/val.bin (200 batches, 819,200 tokens)",
        "50M_pretrain_base": {"loss": 3.1447, "perplexity": 23.21},
        "50M_sft_v1": {"loss": 3.2615, "perplexity": 26.09},
        "50M_sft_v2_best_val": {"loss": 3.3756, "perplexity": 29.24},
        "50M_sft_v2_latest": {"loss": 3.3898, "perplexity": 29.66},
        "loss_drift_vs_base": round(3.3756 - 3.1447, 4)
    },
    "held_out_generalization_val_100": {
        "best_val_step_2400": best_val_data["generalization_val_summary"],
        "latest_step_3225": latest_data["generalization_val_summary"]
    },
    "unseen_train_split_transfer_400": {
        "best_val_step_2400": best_val_data["memorization_train_summary"],
        "latest_step_3225": latest_data["memorization_train_summary"]
    },
    "cross_model_comparison": {
        "python_ast_validity_held_out": {
            "16M_pretrain_base": "3/40 (7.5%)",
            "16M_sft": "31/40 (77.5%)",
            "50M_pretrain_base": "6/40 (15.0%)",
            "50M_sft_v1": "31/40 (77.5%)",
            "50M_sft_v2_best_val": "39/40 (97.5%)",
            "50M_sft_v2_latest": "38/40 (95.0%)"
        },
        "python_functional_pass_held_out": {
            "16M_pretrain_base": "0/40 (0.0%)",
            "16M_sft": "0/40 (0.0%)",
            "50M_pretrain_base": "0/40 (0.0%)",
            "50M_sft_v1": "0/40 (0.0%)",
            "50M_sft_v2_best_val": "0/40 (0.0%)",
            "50M_sft_v2_latest": "1/40 (2.5%)"
        },
        "json_syntax_validity_held_out": {
            "16M_pretrain_base": "0/40 (0.0%)",
            "16M_sft": "27/40 (67.5%)",
            "50M_pretrain_base": "0/40 (0.0%)",
            "50M_sft_v1": "19/40 (47.5%)",
            "50M_sft_v2_best_val": "3/40 (7.5%)",
            "50M_sft_v2_latest": "13/40 (32.5%)"
        },
        "qa_accuracy_held_out": {
            "16M_pretrain_base": "0/10 (0.0%)",
            "16M_sft": "0/10 (0.0%)",
            "50M_pretrain_base": "0/10 (0.0%)",
            "50M_sft_v1": "0/10 (0.0%)",
            "50M_sft_v2_best_val": "0/10 (0.0%)",
            "50M_sft_v2_latest": "0/10 (0.0%)"
        },
        "explanation_distinct_1_held_out": {
            "16M_sft": 0.4034,
            "50M_sft_v1": 0.4637,
            "50M_sft_v2_best_val": 0.4904,
            "50M_sft_v2_latest": 0.3687
        }
    }
}

out_path = Path("artifacts/reports/sft_50m_targeted_v2_metrics.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"[OK] Metrics saved to {out_path}")
