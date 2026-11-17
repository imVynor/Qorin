import json
from pathlib import Path

def create_experiment_registry():
    registry = {
        "metadata": {
            "title": "ModernGPT-S / QORIN Canonical Experiment Registry",
            "version": "1.0.0",
            "last_updated": "2026-11-17T00:45:00Z",
            "institution": "ES 335 Machine Learning Research",
            "project": "Tool-Augmented Small Language Models: 514M ModernGPT-S",
            "architecture_baseline": "ModernGPT-S (514M parameters, 12 layers, 1536 dim, GQA 12/4 heads, RoPE, RMSNorm, SwiGLU, 4096 context)",
            "tokenizer_baseline": "Custom HuggingFace Byte-Pair Encoding (32,768 vocabulary, special tokens: <|im_start|>, <|im_end|>, <|tool_call|>, <|tool_result|>, <|tool_error|>)"
        },
        "experiments": [
            {
                "id": "base_pretrained",
                "name": "ModernGPT-S 500M Pretrained Base",
                "phase": "Phase 1c / Phase 1d",
                "status": "completed_and_frozen",
                "parent_checkpoint": None,
                "checkpoint_path": "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
                "checkpoint_sha256": "28ac47b07e9616967d6a2ea0458e68432682085a2e05fba6ac2e55b82b6d639d",
                "architecture": {
                    "parameters": 514480128,
                    "unique_parameters": 514480128,
                    "layers": 12,
                    "hidden_dim": 1536,
                    "heads": 12,
                    "kv_heads": 4,
                    "context_length": 4096,
                    "vocab_size": 32768,
                    "positional_encoding": "RoPE (Rotary Position Embeddings)",
                    "normalization": "RMSNorm",
                    "activation": "SwiGLU",
                    "tie_embeddings": False
                },
                "training_data": {
                    "dataset_name": "5B Token Multidomain Pretraining Corpus (Curated 1.12B tokens consumed)",
                    "token_count": 1121648640,
                    "domains": {
                        "web": 0.40,
                        "edu": 0.15,
                        "code": 0.15,
                        "wiki": 0.075,
                        "math": 0.075,
                        "books": 0.10,
                        "dialogue": 0.05
                    }
                },
                "training_config": {
                    "optimizer": "AdamW",
                    "learning_rate": 3.0e-4,
                    "precision": "BF16",
                    "gradient_checkpointing": True,
                    "batch_size_tokens": 131072,
                    "total_updates": 8558
                },
                "training_metrics": {
                    "final_training_loss": 2.7410,
                    "final_validation_loss": 2.8621,
                    "final_validation_ppl": 17.50
                },
                "evaluation_metrics": {
                    "pretraining_retention_loss": 2.8621,
                    "pretraining_retention_ppl": 17.50,
                    "coding_syntax_validity_pct": 2.0,
                    "coding_pass_at_1_pct": 0.0,
                    "is_prime_accuracy_pct": 0.0,
                    "mathematics_accuracy_pct": 0.0,
                    "tool_intent_f1": 0.0,
                    "tool_syntax_validity_pct": 0.0,
                    "tool_e2e_success_pct": 4.0,
                    "no_tool_control_accuracy_pct": 100.0,
                    "distinct_1": 0.4877,
                    "repetition_4_rate": 0.3443
                },
                "hardware_telemetry": {
                    "device": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)",
                    "active_compute_hours": 38.4,
                    "cooldown_cycles": 22,
                    "peak_temperature_c": 74.0,
                    "average_tps": 8120
                },
                "known_limitations": [
                    "Base pretraining model lacks instruction-following conditioning (emits completions, echoes, or repetitive continuations)",
                    "Zero tool-intent detection or schema execution",
                    "Zero coding Pass@1 capability"
                ]
            },
            {
                "id": "experiment_a",
                "name": "Experiment A (25M General SFT Pilot)",
                "phase": "Phase 2b",
                "status": "completed_and_frozen",
                "parent_checkpoint": "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
                "checkpoint_path": "modern_gpt/checkpoints/sft_500m_pilot_25m/ckpt_latest.pt",
                "checkpoint_sha256": "249fb467a9cf76e732890cc0501ec39a443756cb2f1c86c66402960f4abbadc4",
                "architecture": {
                    "parameters": 514480128,
                    "vocab_size": 32768,
                    "context_length": 4096
                },
                "training_data": {
                    "dataset_name": "General Instruction Pilot SFT",
                    "token_count": 25000000,
                    "splits": {"train": 0.90, "val": 0.10}
                },
                "training_config": {
                    "optimizer": "8-bit AdamW",
                    "learning_rate": 5.0e-5,
                    "precision": "BF16",
                    "total_updates": 300
                },
                "training_metrics": {
                    "final_validation_loss": 1.6210
                },
                "evaluation_metrics": {
                    "pretraining_retention_loss": 2.8487,
                    "pretraining_retention_ppl": 17.26,
                    "coding_syntax_validity_pct": 52.0,
                    "coding_pass_at_1_pct": 14.0,
                    "is_prime_accuracy_pct": 61.54,
                    "mathematics_accuracy_pct": 0.0,
                    "tool_intent_f1": 0.0,
                    "tool_syntax_validity_pct": 0.0,
                    "tool_e2e_success_pct": 8.0,
                    "no_tool_control_accuracy_pct": 100.0,
                    "distinct_1": 0.4878,
                    "repetition_4_rate": 0.3079
                },
                "hardware_telemetry": {
                    "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
                    "active_compute_hours": 1.45,
                    "peak_temperature_c": 71.2
                },
                "known_limitations": [
                    "0% tool intent recall (no tool training data included)",
                    "Moderate coding syntax validity (52%)",
                    "Low Pass@1 accuracy (14%)"
                ]
            },
            {
                "id": "experiment_b",
                "name": "Experiment B (35M General + Tool-Use SFT)",
                "phase": "Phase 2c",
                "status": "completed_and_frozen",
                "parent_checkpoint": "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
                "checkpoint_path": "modern_gpt/experiments/phase2_experiment_b/stage_4_release/release_checkpoint/ckpt_experiment_b_final.pt",
                "checkpoint_sha256": "acbecc699da76387de9ef6d6a40a64ad73b324b6eee2bd1842a6fd9ab1972c4f",
                "architecture": {
                    "parameters": 514480128,
                    "vocab_size": 32768,
                    "context_length": 4096
                },
                "training_data": {
                    "dataset_name": "General Instruction + Schema Demonstration SFT",
                    "token_count": 35000000,
                    "authentic_tool_tokens": 241565,
                    "authentic_tool_pct": 0.69
                },
                "training_config": {
                    "optimizer": "8-bit AdamW",
                    "learning_rate": 4.0e-5,
                    "precision": "BF16",
                    "total_updates": 450
                },
                "training_metrics": {
                    "final_validation_loss": 1.4883
                },
                "evaluation_metrics": {
                    "pretraining_retention_loss": 2.8499,
                    "pretraining_retention_ppl": 17.29,
                    "coding_syntax_validity_pct": 92.0,
                    "coding_pass_at_1_pct": 42.0,
                    "is_prime_accuracy_pct": 46.15,
                    "mathematics_accuracy_pct": 4.0,
                    "tool_intent_f1": 0.0,
                    "tool_syntax_validity_pct": 0.0,
                    "tool_e2e_success_pct": 4.0,
                    "no_tool_control_accuracy_pct": 100.0,
                    "distinct_1": 0.5075,
                    "repetition_4_rate": 0.2933
                },
                "hardware_telemetry": {
                    "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
                    "active_compute_hours": 2.12,
                    "peak_temperature_c": 72.8
                },
                "known_limitations": [
                    "0% autonomous tool-intent recall: model memorized static schemas as conversational text instead of actionable dispatch triggers",
                    "Only 0.69% authentic execution trajectories in dataset, leading to severe dilution"
                ]
            },
            {
                "id": "experiment_c",
                "name": "Experiment C (Targeted Autonomous Two-Tool SFT)",
                "phase": "Phase 2d",
                "status": "completed_and_verified",
                "parent_checkpoint": "modern_gpt/experiments/phase2_experiment_b/stage_4_release/release_checkpoint/ckpt_experiment_b_final.pt",
                "checkpoint_path": "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt",
                "checkpoint_sha256": "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b",
                "architecture": {
                    "parameters": 514480128,
                    "vocab_size": 32768,
                    "context_length": 4096
                },
                "training_data": {
                    "dataset_name": "Targeted Authentic Two-Tool Learning Dataset (Stage 1)",
                    "dataset_sha256": "5bc837e2ba6e1b6f00db1490214a1f5f24213d2bb0a0aa1570d276b176d6c1ee",
                    "token_count": 1785635,
                    "conversations": 755,
                    "composition": {
                        "python_calculations": 260,
                        "data_transformations": 185,
                        "multistep_workflows": 110,
                        "error_recovery_trajectories": 70,
                        "negative_tool_controls": 80,
                        "web_research_fixtures": 50
                    }
                },
                "training_config": {
                    "optimizer": "8-bit AdamW (Config A)",
                    "learning_rate": 2.0e-5,
                    "min_learning_rate": 2.0e-6,
                    "lr_scheduler": "cosine",
                    "warmup_updates": 15,
                    "precision": "BF16",
                    "gradient_checkpointing": True,
                    "total_updates": 495,
                    "batch_size_sequences": 2,
                    "gradient_accumulation_steps": 4
                },
                "training_metrics": {
                    "initial_train_loss": 1.9169,
                    "final_train_loss": 1.1704,
                    "initial_val_loss": 1.8420,
                    "final_val_loss": 1.1685
                },
                "evaluation_metrics": {
                    "pretraining_retention_loss": 2.8651,
                    "pretraining_retention_ppl": 17.55,
                    "coding_syntax_validity_pct": 96.0,
                    "coding_pass_at_1_pct": 38.0,
                    "is_prime_accuracy_pct": 53.85,
                    "mathematics_accuracy_pct": 0.0,
                    "tool_intent_precision": 1.0,
                    "tool_intent_recall": 0.80,
                    "tool_intent_f1": 0.8889,
                    "tool_syntax_validity_pct": 66.67,
                    "tool_e2e_success_pct": 6.67,
                    "no_tool_control_accuracy_pct": 100.0,
                    "distinct_1": 0.5067,
                    "repetition_4_rate": 0.2951
                },
                "hardware_telemetry": {
                    "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
                    "active_compute_hours": 0.98,
                    "peak_vram_mb": 4210.0,
                    "peak_temperature_c": 70.4,
                    "cooldown_duration_minutes": 15.0
                },
                "known_limitations": [
                    "Strong inductive bias toward Python tool (calls Python on web queries when both tools are active unless explicitly guided)",
                    "Semantic code bugs in generated snippets: model generates valid Python syntax that runs, but occasionally computes wrong algebraic expressions",
                    "Round 2 interpretation is faithful to tool outputs, but lacks mathematical verification or error reflection"
                ]
            }
        ]
    }

    out_file = Path("modern_gpt/reports/final_research_audit/experiment_registry.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    create_experiment_registry()
