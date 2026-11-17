import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "modern_gpt"))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.conversation_serializer import ConversationSerializer

def analyze_training_log(log_path: str):
    p = Path(log_path)
    if not p.exists():
        return None

    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return None

    steps = [r["global_step"] for r in records]
    losses = [r["train_loss"] for r in records]
    grad_norms = [r["grad_norm"] for r in records]
    lrs = [r["learning_rate"] for r in records]
    vram = [r["allocated_vram_mb"] for r in records]

    # Group by epoch
    epochs = sorted(list(set(r.get("epoch", 1) for r in records)))
    epoch_summaries = {}
    for ep in epochs:
        ep_recs = [r for r in records if r.get("epoch") == ep]
        ep_losses = [r["train_loss"] for r in ep_recs]
        ep_gnorms = [r["grad_norm"] for r in ep_recs]
        epoch_summaries[f"epoch_{ep}"] = {
            "steps": len(ep_recs),
            "start_step": ep_recs[0]["global_step"],
            "end_step": ep_recs[-1]["global_step"],
            "initial_loss": ep_losses[0],
            "final_loss": ep_losses[-1],
            "min_loss": min(ep_losses),
            "max_loss": max(ep_losses),
            "avg_loss": round(sum(ep_losses) / len(ep_losses), 4),
            "avg_grad_norm": round(sum(ep_gnorms) / len(ep_gnorms), 4),
            "max_grad_norm": max(ep_gnorms),
        }

    return {
        "total_records": len(records),
        "total_optimizer_steps": steps[-1],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "min_loss_step": steps[losses.index(min(losses))],
        "max_loss": max(losses),
        "avg_loss": round(sum(losses) / len(losses), 4),
        "peak_lr": max(lrs),
        "final_lr": lrs[-1],
        "avg_grad_norm": round(sum(grad_norms) / len(grad_norms), 4),
        "max_grad_norm": max(grad_norms),
        "avg_vram_mb": round(sum(vram) / len(vram), 1),
        "epochs": epoch_summaries,
    }

def verify_loss_masking():
    tok = HuggingFaceBPEWrapper.load("modern_gpt/artifacts/tokenizer_16k_v2")
    serializer = ConversationSerializer(tok)

    test_conversations = [
        # Standard 1-turn
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 is 4."}
        ],
        # Multi-turn
        [
            {"role": "system", "content": "You are ModernGPT-S."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Nice to meet you, Bob!"}
        ],
        # Missing system turn (auto-injected)
        [
            {"role": "user", "content": "Say test."},
            {"role": "assistant", "content": "test"}
        ]
    ]

    masking_results = []
    for idx, msgs in enumerate(test_conversations):
        enc = serializer.encode_conversation(msgs)
        input_ids = enc["input_ids"].tolist()
        labels = enc["labels"].tolist()

        # Check: Every token corresponding to system or user must be -100 in labels
        # Every token corresponding to assistant content and closing must be token_id
        # Let's verify per-turn tokens
        tokens_detail = []
        for inp, lbl in zip(input_ids, labels):
            tokens_detail.append({
                "id": inp,
                "text": tok.decode([inp]),
                "label": lbl,
                "supervised": lbl != -100
            })

        masking_results.append({
            "test_idx": idx,
            "total_tokens": len(input_ids),
            "supervised_tokens": (enc["labels"] != -100).sum().item(),
            "first_15_labels": labels[:15],
            "last_10_labels": labels[-10:],
            "contains_masked_system": labels[0] == -100,
            "eos_supervised": labels[-1] == tok.eos_token_id,
        })

    return masking_results

def main():
    print("[*] Running Training Audit...")
    log_8d = "modern_gpt/artifacts/experiments/modern_gpts_100m_step_08d/step_08d_training_log.jsonl"
    log_7b = "modern_gpt/artifacts/experiments/modern_gpts_100m_step_07b/step_07b_training_log.jsonl"

    res_8d = analyze_training_log(log_8d)
    res_7b = analyze_training_log(log_7b)

    masking_res = verify_loss_masking()

    audit_summary = {
        "step_08d_log_summary": res_8d,
        "step_07b_log_summary": res_7b,
        "masking_verification": masking_res,
    }

    out_file = Path("modern_gpt/diagnostics/phase0/training_audit_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"[✓] Saved training audit results to {out_file}")

if __name__ == "__main__":
    main()

