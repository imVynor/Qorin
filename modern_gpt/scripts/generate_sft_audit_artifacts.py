import json
from pathlib import Path

out_dir = Path(r"c:\Users\Anish\Desktop\Projects\ML-course\modern_gpt\artifacts\experiments\modern_gpts_100m_sft_audit")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. checkpoint_integrity.json
ckpt_integ = {
    "target_starting_checkpoint": {
        "path": "checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt",
        "sha256": "3b23f37b82279c63d5913933fef22ea487ed28f28e08878cfa4d531064900817",
        "step": 2000,
        "tokens_seen": 65536000,
        "parameters": 100401408,
        "status": "VERIFIED_VALID_STARTING_POINT"
    },
    "preserved_checkpoints": {
        "original_100m_best_val": {
            "path": "checkpoints/modern_gpts_100m/best_val.pt",
            "sha256": "bded509c44eabc2932dc1e293967bb0524acc31182fe9df0295b9ad2a4671e31",
            "status": "VERIFIED_UNTOUCHED"
        },
        "init_4k": {
            "path": "checkpoints/modern_gpts_100m_4k/init_4k.pt",
            "sha256": "c307c4304ad3db647a09cc4b90647714c97257dab2e3d53efe8683cc0b5c1b96",
            "status": "VERIFIED_UNTOUCHED"
        }
    },
    "tokenizer": {
        "path": "artifacts/tokenizer_16k_v2",
        "fingerprint": "be6c3d60c5baf217",
        "vocab_size": 16000
    }
}
with open(out_dir / "checkpoint_integrity.json", "w", encoding="utf-8") as f:
    json.dump(ckpt_integ, f, indent=2)

# 2. sft_compatibility_matrix.json
compat = {
    "model_loading": {
        "current_state": "Hardcoded to 50M/16M in train_sft.py (checks emb_dim == 512)",
        "100m_compatibility": "Incompatible without update (100M has emb_dim=768, n_heads=12, num_kv_heads=4)",
        "required_action": "Update train_sft.py to load configuration directly from checkpoint model_config or get_100m_gqa4_4k_config()",
        "status": "REQUIRES_MODIFICATION"
    },
    "gqa_architecture": {
        "current_state": "train_sft.py instantiates ModelConfig without num_kv_heads (defaults to MHA)",
        "100m_compatibility": "Causes shape mismatch on key/value projection weights (4 KV heads vs 12 heads)",
        "required_action": "Pass num_kv_heads=4 to ModelConfig in train_sft.py",
        "status": "REQUIRES_MODIFICATION"
    },
    "rope_4k": {
        "current_state": "RotaryEmbedding accepts rope_scale and rope_type, but train_sft.py omitted them",
        "100m_compatibility": "Would reset rope_scale to 1.0 (512 baseline)",
        "required_action": "Pass rope_scale=8.0, rope_type='linear', context_length=4096 in train_sft.py",
        "status": "REQUIRES_MODIFICATION"
    },
    "tokenization": {
        "current_state": "HuggingFaceBPEWrapper load and encode working properly",
        "100m_compatibility": "Fully compatible with active bpe_16k_v2",
        "required_action": "None",
        "status": "READY"
    },
    "multi_turn_serialization": {
        "current_state": "train_sft.py only formats single-turn Alpaca prompts",
        "100m_compatibility": "Cannot train multi-turn conversational agents or tool interaction",
        "required_action": "Implement ChatML or standardized multi-turn message serializer with role tokens",
        "status": "REQUIRES_MODIFICATION"
    },
    "assistant_loss_masking": {
        "current_state": "Masks prompt tokens with -100 for single turn; splits input/output",
        "100m_compatibility": "Works for single turn, but lacks turn-by-turn masking for multi-turn conversations",
        "required_action": "Implement token-level role-based masking: only assistant messages have non -100 labels",
        "status": "REQUIRES_MODIFICATION"
    },
    "padding_and_attention": {
        "current_state": "collate_sft_batch pads to max batch length with right-padding and -100 labels",
        "100m_compatibility": "Compatible, causal SDPA handles right-padding correctly",
        "required_action": "Keep right padding, enforce 4096 max sequence length cap",
        "status": "READY"
    },
    "gradient_accumulation": {
        "current_state": "Implemented in train_sft.py with loss division",
        "100m_compatibility": "Compatible",
        "required_action": "Set microbatch 1, grad accum 8 (32,768 tokens per update)",
        "status": "READY"
    },
    "checkpoint_recovery": {
        "current_state": "Saves latest.pt and best_val.pt with basic optimizer state",
        "100m_compatibility": "Partially complete, lacks RNG states and session guardrails from pretrain_100m.py",
        "required_action": "Port complete 19-point state recovery and GPUMonitor thermal guardrails",
        "status": "REQUIRES_MODIFICATION"
    },
    "evaluation": {
        "current_state": "Calculates SFT validation loss and has AST/JSON evaluators",
        "100m_compatibility": "Existing evaluators assume 512 context and single-turn format",
        "required_action": "Integrate multi-length validation (512-4K) and dedicated long-context retrieval benchmark",
        "status": "REQUIRES_MODIFICATION"
    }
}
with open(out_dir / "sft_compatibility_matrix.json", "w", encoding="utf-8") as f:
    json.dump(compat, f, indent=2)

# 3. existing_dataset_inventory.json
inv = {
    "sft_v2_public": {
        "path": "data/sft_v2_public/train.jsonl",
        "total_examples": 8597,
        "total_tokens": 1939904,
        "max_sequence_length": 512,
        "avg_sequence_length": 225.6,
        "multi_turn": False,
        "long_context": False,
        "sources": ["mbpp", "code_alpaca", "dolly_instruction", "dolly_qa", "ultrachat", "squad", "gsm8k", "structured_json"],
        "suitability_for_4k_sft": "Suitable for short-context instruction anchor, but completely lacks >512 context and multi-turn turns"
    },
    "midscale_500": {
        "path": "data/midscale_500/train.jsonl",
        "total_examples": 500,
        "suitability_for_4k_sft": "Deprecated legacy v1 dataset; severe contamination with transfer benchmark"
    }
}
with open(out_dir / "existing_dataset_inventory.json", "w", encoding="utf-8") as f:
    json.dump(inv, f, indent=2)

# 4. proposed_dataset_composition.json
prop_data = {
    "target_total_examples": 12000,
    "target_total_tokens": 12000000,
    "training_epochs": 3,
    "total_training_tokens": 36000000,
    "categories": {
        "general_instruction": {
            "purpose": "Follow user tasks and direct requests",
            "examples": 2000,
            "token_range": "64 - 512",
            "source": "sft_v2_public (filtered dolly/ultrachat)",
            "share_pct": 16.7
        },
        "multi_turn_dialogue": {
            "purpose": "Multi-turn conversational continuity and reference tracking",
            "examples": 2500,
            "token_range": "256 - 2048",
            "source": "UltraChat 200k (multi-turn subset)",
            "share_pct": 20.8
        },
        "educational_explanations": {
            "purpose": "Detailed concept explanations, STEM tutorials",
            "examples": 1500,
            "token_range": "256 - 1024",
            "source": "OpenAssistant / Dolly educational subsets",
            "share_pct": 12.5
        },
        "coding_and_python": {
            "purpose": "Python generation, bug fixing, explanations",
            "examples": 2000,
            "token_range": "128 - 1024",
            "source": "MBPP (verified) + CodeAlpaca (AST verified)",
            "share_pct": 16.7
        },
        "structured_json": {
            "purpose": "Strict JSON schema formatting and key-value extraction",
            "examples": 800,
            "token_range": "128 - 512",
            "source": "Synthetic JSON schema extraction",
            "share_pct": 6.7
        },
        "math_reasoning": {
            "purpose": "Step-by-step arithmetic and logic word problems",
            "examples": 700,
            "token_range": "128 - 512",
            "source": "GSM8K (train split only)",
            "share_pct": 5.8
        },
        "contextual_qa": {
            "purpose": "Direct QA from short context documents",
            "examples": 1000,
            "token_range": "256 - 1024",
            "source": "SQuAD v2 (train split only)",
            "share_pct": 8.3
        },
        "long_context_retrieval": {
            "purpose": "Single and multi-needle retrieval across 1K-4K context",
            "examples": 1000,
            "token_range": "1024 - 4096",
            "source": "Synthetic Long-Context Retrieval Generator",
            "share_pct": 8.3
        },
        "long_document_understanding": {
            "purpose": "Summarization and synthesis over extended context",
            "examples": 500,
            "token_range": "2048 - 4096",
            "source": "BookSum / QASPER / GovReport filtered subsets",
            "share_pct": 4.2
        }
    }
}
with open(out_dir / "proposed_dataset_composition.json", "w", encoding="utf-8") as f:
    json.dump(prop_data, f, indent=2)

# 5. retrieval_curriculum.json
retrieval = {
    "curriculum_stages": [
        {
            "stage": 1,
            "name": "Short-Context Exact Fact Retrieval",
            "context_length": "256 - 512",
            "depths": ["10%", "50%", "90%"],
            "distractor_density": "Low"
        },
        {
            "stage": 2,
            "name": "Medium-Context Position Robustness",
            "context_length": "1024 - 2048",
            "depths": ["5%", "25%", "50%", "75%", "95%"],
            "distractor_density": "Medium"
        },
        {
            "stage": 3,
            "name": "Long-Context Multi-Needle Retrieval",
            "context_length": "2048 - 4096",
            "depths": ["Multiple sections simultaneously"],
            "distractor_density": "High"
        },
        {
            "stage": 4,
            "name": "Multi-Turn Conversational Reference & Resolution",
            "context_length": "1024 - 4096",
            "depths": ["Turn 1 reference tested at Turn 5"],
            "distractor_density": "Conversational filler"
        },
        {
            "stage": 5,
            "name": "Document-Grounded Reasoning & Unknown Handling",
            "context_length": "2048 - 4096",
            "depths": ["Synthesize 2 facts; or express 'Cannot be determined' when absent"],
            "distractor_density": "Real world technical text"
        }
    ],
    "mixture_policy": "Interleaved mixture (joint training from Step 0 to prevent catastrophic forgetting of short-context capabilities)"
}
with open(out_dir / "retrieval_curriculum.json", "w", encoding="utf-8") as f:
    json.dump(retrieval, f, indent=2)

# 6. gpu_feasibility_estimate.json
gpu_est = {
    "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "total_vram_gb": 7.93,
    "static_memory_gb": 1.57,
    "measured_sft_microbatch_1_4096": {
        "allocated_vram_gb": 3.41,
        "reserved_vram_gb": 3.66,
        "headroom_gb": 4.27,
        "utilization_pct": 46.2
    },
    "training_time_projections": {
        "effective_tokens_per_step": 32768,
        "gradient_accumulation": 8,
        "estimated_step_time_ms": 2200,
        "tokens_per_sec": 14890,
        "total_tokens_3_epochs": 36000000,
        "total_steps": 1100,
        "estimated_active_training_hours": 0.67
    }
}
with open(out_dir / "gpu_feasibility_estimate.json", "w", encoding="utf-8") as f:
    json.dump(gpu_est, f, indent=2)

# 7. audit_summary.json
summ = {
    "audit_target": "ModernGPT-S 100M GQA-4 SFT at 4K Context",
    "base_checkpoint_status": "VERIFIED_VALID_STARTING_POINT (Pilot Step 2000, Loss 2.6997 at 4K)",
    "readiness_classification": "READY_AFTER_MODIFICATIONS",
    "key_bottlenecks_identified": [
        "train_sft.py hardcoded to 50M/16M and context_length=512",
        "train_sft.py lacks num_kv_heads (GQA) and rope_scale/rope_type",
        "Dataset serialization in train_sft.py is single-turn Alpaca only (no multi-turn or role masking)",
        "Existing sft_v2_public dataset is strictly <=512 tokens (zero 4K examples)"
    ],
    "recommended_next_actions": [
        "Phase A: Adapt train_sft.py for 100M GQA-4, 4K RoPE, and ChatML token-level loss masking",
        "Phase B: Construct 4K SFT dataset mixture including multi-turn dialogue and explicit synthetic retrieval",
        "Phase C: Execute controlled 500-step SFT pilot on RTX 5060 with thermal monitoring"
    ]
}
with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=2)

print("All 7 audit artifacts generated successfully in:", out_dir)

