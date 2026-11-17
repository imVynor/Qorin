"""Populate historical ModernGPT-S experiments into the experiment tracking registry.
Registers:
1. EXP-PRETRAIN-16M-100M: 16M Pretrained Baseline
2. EXP-SFT-16M-MIDSCALE-500: 16M SFT Baseline
3. EXP-PRETRAIN-50M-250M: 50M Pretrained Baseline
4. EXP-SFT-50M-MIDSCALE-500: 50M SFT v1
5. EXP-SFT-50M-TARGETED-V2-BEST: 50M SFT v2 Best Val Checkpoint
6. EXP-SFT-50M-TARGETED-V2-LATEST: 50M SFT v2 Latest Checkpoint
7. EXP-AUDIT-SFT-50M-V2: SFT v2 Diagnostic Audit
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.registry import ExperimentRegistry
from experiments.tracker import compute_file_sha256
import json
import yaml


def register_all():
    exp_dir = Path(__file__).parent
    registry = ExperimentRegistry(exp_dir)

    historical_records = [
        {
            "experiment_id": "EXP-PRETRAIN-16M-100M",
            "experiment_name": "ModernGPT-S 16M Pretraining Baseline",
            "category": "pretraining",
            "architecture": "ModernGPT-16M (6 layers, 6 heads, 384 dim, 1024 ffn)",
            "parameter_count": 16252800,
            "dataset": "artifacts/shards/bpe_16k_v2/",
            "dataset_version": "v2",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 102300000,
            "training_config": {
                "batch_size": 16,
                "grad_accum": 4,
                "context_length": 512,
                "learning_rate": 4e-4,
                "precision": "bf16",
                "scheduler": "cosine",
            },
            "start_time": "2026-10-27 18:00:00 UTC",
            "completion_time": "2026-10-28 01:00:00 UTC",
            "status": "completed",
            "checkpoint_locations": ["checkpoints/exp_100m/best_val.pt", "checkpoints/exp_100m/latest.pt"],
            "best_val_metric": 3.8615,
            "final_val_metric": 3.8615,
            "evaluation_report_location": "artifacts/reports/evaluation_100m_report.json",
            "experiment_report_location": "artifacts/reports/evaluation_100m_report.md",
            "parent_experiment_id": "None",
            "main_conclusion": "16M model successfully converged on 100M tokens; established the reference small-scale language model baseline (val loss 3.8615, PPL 47.54).",
        },
        {
            "experiment_id": "EXP-SFT-16M-MIDSCALE-500",
            "experiment_name": "ModernGPT-S 16M SFT Baseline",
            "category": "SFT",
            "architecture": "ModernGPT-16M (6 layers, 6 heads, 384 dim, 1024 ffn)",
            "parameter_count": 16252800,
            "dataset": "data/midscale_500/train.jsonl",
            "dataset_version": "midscale_500",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 90000,
            "training_config": {
                "epochs": 5.0,
                "batch_size": 2,
                "grad_accum": 4,
                "learning_rate": 5e-5,
                "precision": "bf16",
            },
            "start_time": "2026-10-29 02:00:00 UTC",
            "completion_time": "2026-10-29 02:30:00 UTC",
            "status": "completed",
            "checkpoint_locations": ["checkpoints/sft_midscale_500/best_val.pt"],
            "best_val_metric": 1.52,
            "final_val_metric": 1.52,
            "evaluation_report_location": "artifacts/diagnostics/midscale_500_evaluation.json",
            "experiment_report_location": "artifacts/reports/sft_v2_evaluation_report.json",
            "parent_experiment_id": "EXP-PRETRAIN-16M-100M",
            "main_conclusion": "SFT leaped Python AST validity from 7.5% to 77.5% and JSON from 0.0% to 67.5%, but achieved 0.0% functional test correctness on held-out tasks.",
        },
        {
            "experiment_id": "EXP-PRETRAIN-50M-250M",
            "experiment_name": "ModernGPT-S 50M Pretraining Baseline",
            "category": "pretraining",
            "architecture": "ModernGPT-50M (12 layers, 8 heads, 512 dim, 1536 ffn)",
            "parameter_count": 49105408,
            "dataset": "artifacts/shards_250m/",
            "dataset_version": "250m",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 250019840,
            "training_config": {
                "batch_size": 16,
                "grad_accum": 4,
                "context_length": 512,
                "learning_rate": 3e-4,
                "min_lr": 3e-5,
                "precision": "bf16",
                "warmup_steps": 500,
                "decay_steps": 7630,
            },
            "start_time": "2026-11-02 22:00:00 UTC",
            "completion_time": "2026-11-03 01:30:00 UTC",
            "status": "completed",
            "checkpoint_locations": ["checkpoints/pretrain_50m_250m/best_val.pt", "checkpoints/pretrain_50m_250m/latest.pt"],
            "best_val_metric": 3.1169,
            "final_val_metric": 3.1152,
            "evaluation_report_location": "artifacts/reports/capability_16m_vs_50m_comparison.json",
            "experiment_report_location": "artifacts/reports/50m_vs_16m_scaling_and_capability_evaluation.md",
            "parent_experiment_id": "EXP-PRETRAIN-16M-100M",
            "main_conclusion": "50M capacity unlocked 52.6% perplexity reduction over 16M (PPL 22.54 vs 47.54), demonstrating healthy scaling. Pretraining alone did not unlock task execution (0% functional).",
        },
        {
            "experiment_id": "EXP-SFT-50M-MIDSCALE-500",
            "experiment_name": "ModernGPT-S 50M SFT v1",
            "category": "SFT",
            "architecture": "ModernGPT-50M (12 layers, 8 heads, 512 dim, 1536 ffn)",
            "parameter_count": 49105408,
            "dataset": "data/midscale_500/train.jsonl",
            "dataset_version": "midscale_500",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 90000,
            "training_config": {
                "epochs": 5.0,
                "batch_size": 2,
                "grad_accum": 4,
                "learning_rate": 5e-5,
                "precision": "bf16",
            },
            "start_time": "2026-11-03 01:45:00 UTC",
            "completion_time": "2026-11-03 01:50:00 UTC",
            "status": "completed",
            "checkpoint_locations": ["checkpoints/sft_50m_midscale_500/latest.pt"],
            "best_val_metric": 1.48,
            "final_val_metric": 1.48,
            "evaluation_report_location": "artifacts/diagnostics/sft_50m_midscale_500_evaluation.json",
            "experiment_report_location": "artifacts/reports/sft_50m_midscale_500_results.md",
            "parent_experiment_id": "EXP-PRETRAIN-50M-250M",
            "main_conclusion": "Synthetic 400-example SFT fine-tuned output syntax (77.5% AST) but memorized templates without achieving generalization to held-out functional tests (0.0%).",
        },
        {
            "experiment_id": "EXP-SFT-50M-TARGETED-V2-BEST",
            "experiment_name": "ModernGPT-S 50M SFT v2 Best Val Checkpoint",
            "category": "SFT",
            "architecture": "ModernGPT-50M (12 layers, 8 heads, 512 dim, 1536 ffn)",
            "parameter_count": 49105408,
            "dataset": "data/sft_v2_public/train.jsonl",
            "dataset_version": "sft_v2_public",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 1939120,
            "training_config": {
                "epochs": 3.0,
                "batch_size": 2,
                "grad_accum": 4,
                "learning_rate": 2.5e-5,
                "min_lr": 2.5e-6,
                "warmup_steps": 100,
                "precision": "bf16",
            },
            "start_time": "2026-11-03 02:52:00 UTC",
            "completion_time": "2026-11-03 03:01:00 UTC",
            "status": "completed",
            "checkpoint_locations": ["checkpoints/sft_50m_targeted_v2/best_val.pt"],
            "best_val_metric": 1.4713,
            "final_val_metric": 1.4713,
            "evaluation_report_location": "artifacts/diagnostics/sft_50m_targeted_v2_best_val_eval.json",
            "experiment_report_location": "artifacts/reports/sft_50m_targeted_v2_results.md",
            "parent_experiment_id": "EXP-PRETRAIN-50M-250M",
            "main_conclusion": "Reached lowest selection validation loss (1.4713) and 97.5% AST validity, but suffered from 60% Python wrapper bias on JSON and 0.0% held-out functional pass.",
        },
        {
            "experiment_id": "EXP-SFT-50M-TARGETED-V2-LATEST",
            "experiment_name": "ModernGPT-S 50M SFT v2 Latest Checkpoint",
            "category": "SFT",
            "architecture": "ModernGPT-50M (12 layers, 8 heads, 512 dim, 1536 ffn)",
            "parameter_count": 49105408,
            "dataset": "data/sft_v2_public/train.jsonl",
            "dataset_version": "sft_v2_public",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 5817360,
            "training_config": {
                "epochs": 3.0,
                "total_steps": 3225,
                "batch_size": 2,
                "grad_accum": 4,
                "learning_rate": 2.5e-5,
                "min_lr": 2.5e-6,
                "precision": "bf16",
            },
            "start_time": "2026-11-03 02:52:00 UTC",
            "completion_time": "2026-11-03 03:01:34 UTC",
            "status": "completed",
            "checkpoint_locations": ["checkpoints/sft_50m_targeted_v2/latest.pt"],
            "best_val_metric": 1.4713,
            "final_val_metric": 1.4788,
            "evaluation_report_location": "artifacts/diagnostics/sft_50m_targeted_v2_latest_eval.json",
            "experiment_report_location": "artifacts/reports/sft_50m_targeted_v2_results.md",
            "parent_experiment_id": "EXP-PRETRAIN-50M-250M",
            "main_conclusion": "Achieved project's first held-out functional Python test pass (2.5%) and 22.5% raw JSON syntax. Functionally superior checkpoint over best_val.pt.",
        },
        {
            "experiment_id": "EXP-AUDIT-SFT-50M-V2",
            "experiment_name": "ModernGPT-S 50M SFT v2 Diagnostic Audit",
            "category": "audit",
            "architecture": "ModernGPT-50M",
            "parameter_count": 49105408,
            "dataset": "data/midscale_500/val.jsonl",
            "dataset_version": "val_100",
            "tokenizer_fingerprint": "be6c3d60c5baf217",
            "training_token_budget": 0,
            "training_config": {
                "mode": "controlled_audit",
                "benchmarks": ["held_out_100", "transfer_160", "pretrain_val_shard"],
            },
            "start_time": "2026-11-03 18:20:00 UTC",
            "completion_time": "2026-11-03 18:40:00 UTC",
            "status": "completed",
            "checkpoint_locations": [
                "checkpoints/pretrain_50m_250m/latest.pt",
                "checkpoints/sft_50m_targeted_v2/best_val.pt",
                "checkpoints/sft_50m_targeted_v2/latest.pt"
            ],
            "best_val_metric": 1.4713,
            "final_val_metric": 1.4788,
            "evaluation_report_location": "artifacts/diagnostics/sft_v2_audit/sft_v2_checkpoint_comparison.json",
            "experiment_report_location": "artifacts/reports/sft_50m_targeted_v2_diagnostic_audit.md",
            "parent_experiment_id": "EXP-SFT-50M-TARGETED-V2-LATEST",
            "main_conclusion": "Proved zero data leakage, verified latest.pt functional superiority, classified 46% of Python failures as function signature mismatches, identified JSON template trap, and measured 27.7% pretrain perplexity increase.",
        }
    ]

    for rec in historical_records:
        registry.register_or_update(rec)
        # Create run directory structure
        run_dir = exp_dir / "runs" / rec["experiment_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation").mkdir(exist_ok=True)
        (run_dir / "profiling").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        # Write manifest.json
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        # Write config.yaml
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(rec["training_config"], f)

    print(f"[+] Successfully registered {len(historical_records)} historical experiments.")

if __name__ == "__main__":
    register_all()

