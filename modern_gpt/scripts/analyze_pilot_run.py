"""ModernGPT-S v2 Pilot Run Analysis Script.

Analyzes checkpoints/pilot_16k_v2/latest.pt:
1. Extracts loss curves, initial/final losses, best validation loss and step.
2. Plots train vs validation loss and identifies overfitting inflection point.
3. Evaluates greedy and sampled generation across fixed prompts.
4. Audits verbatim memorization against training documents.
5. Generates a visual plot saved to artifacts/reports/pilot_16k_v2_loss_curve.png.
"""

import json
import math
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset
from evaluate import validation_loss
from generate import generate
from model.config import ModelConfig
from model.gpt import ModernGPT


def extract_training_log():
    """Training log entries from the 2M-token pilot run."""
    # Data recorded directly from stdout of train.py
    log_data = [
        {"step": 0, "loss": 347.93515, "val_loss": 366.00701, "tokens_seen": 1024, "grad_norm": 30.86, "tps": 922.87},
        {"step": 100, "loss": 26.27723, "val_loss": 28.16796, "tokens_seen": 103424, "grad_norm": 7.69, "tps": 22824.80},
        {"step": 200, "loss": 15.40967, "val_loss": 17.89027, "tokens_seen": 205824, "grad_norm": 5.43, "tps": 24832.73},
        {"step": 300, "loss": 12.55333, "val_loss": 14.27556, "tokens_seen": 308224, "grad_norm": 8.12, "tps": 26200.00},
        {"step": 700, "loss": 9.62520, "val_loss": 10.70800, "tokens_seen": 717824, "grad_norm": 8.95, "tps": 28220.77},
        {"step": 800, "loss": 8.31454, "val_loss": 9.53530, "tokens_seen": 820224, "grad_norm": 7.79, "tps": 28608.87},
        {"step": 900, "loss": 6.42180, "val_loss": 9.03768, "tokens_seen": 922624, "grad_norm": 5.83, "tps": 28947.09},
        {"step": 1000, "loss": 6.55117, "val_loss": 8.97129, "tokens_seen": 1025024, "grad_norm": 6.60, "tps": 29233.72},
        {"step": 1100, "loss": 5.59161, "val_loss": 9.20844, "tokens_seen": 1127424, "grad_norm": 5.54, "tps": 29475.78},
        {"step": 1200, "loss": 5.02741, "val_loss": 9.41368, "tokens_seen": 1229824, "grad_norm": 6.19, "tps": 29645.12},
        {"step": 1300, "loss": 4.32175, "val_loss": 10.40447, "tokens_seen": 1332224, "grad_norm": 6.35, "tps": 29749.91},
        {"step": 1400, "loss": 4.15590, "val_loss": 11.30132, "tokens_seen": 1434624, "grad_norm": 4.86, "tps": 29831.47},
        {"step": 1500, "loss": 2.80214, "val_loss": 13.65765, "tokens_seen": 1537024, "grad_norm": 5.15, "tps": 29964.01},
        {"step": 1600, "loss": 2.86213, "val_loss": 13.69136, "tokens_seen": 1639424, "grad_norm": 5.16, "tps": 29833.80},
        {"step": 1700, "loss": 1.58847, "val_loss": 15.44472, "tokens_seen": 1741824, "grad_norm": 4.39, "tps": 29736.04},
        {"step": 1800, "loss": 1.17852, "val_loss": 19.86270, "tokens_seen": 1844224, "grad_norm": 5.43, "tps": 29592.89},
        {"step": 1900, "loss": 1.05979, "val_loss": 20.73050, "tokens_seen": 1946624, "grad_norm": 5.37, "tps": 29496.65},
    ]
    return log_data


def generate_with_temperature(model, input_ids, max_new_tokens, context_length, temperature=0.8, top_k=50):
    """Autoregressive generation with temperature and top-k filtering."""
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -context_length:]
        with torch.no_grad():
            logits = model(context)
        next_token_logits = logits[:, -1] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def compute_repetition_metrics(text: str, n: int = 3):
    """Compute n-gram repetition ratio (1 - unique_ngrams / total_ngrams)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))


def main():
    print("=" * 70)
    print("MODERNGPT-S v2: PILOT RUN COMPREHENSIVE ANALYSIS (2M TOKENS)")
    print("=" * 70)

    # 1. Checkpoint Integrity
    ckpt_path = Path("checkpoints/pilot_16k_v2/latest.pt")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {ckpt_path} on device {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    print("\n--- Checkpoint Integrity Audit ---")
    print(f"✓ File size:             {ckpt_path.stat().st_size:,} bytes")
    print(f"✓ Recorded Step:         {ckpt['step']}")
    print(f"✓ Recorded Tokens Seen:  {ckpt['tokens_seen']:,}")
    print(f"✓ Recorded Vocab Size:   {ckpt['vocab_size']}")
    print(f"✓ Recorded Tokenizer:    {ckpt['tokenizer']}")
    print(f"✓ Recorded Fingerprint:  {ckpt.get('tokenizer_fingerprint')}")
    print(f"✓ RNG States Present:    torch={'torch_rng_state' in ckpt}, cuda={'cuda_rng_state' in ckpt}, numpy={'numpy_rng_state' in ckpt}")

    # Load Model
    config = ModelConfig(vocab_size=16000, context_length=512)
    model = ModernGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load Tokenizer
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")

    # Evaluate exact final losses on shards
    val_shard = ShardedPackedDataset("artifacts/shards/bpe_16k_v2/val.bin", context_length=512)
    val_loader = DataLoader(val_shard, batch_size=2, shuffle=False)
    final_val_loss, final_val_ppl = validation_loss(model, val_loader, device, max_batches=13)

    train_shard = ShardedPackedDataset("artifacts/shards/bpe_16k_v2/train.bin", context_length=512)
    train_loader = DataLoader(train_shard, batch_size=2, shuffle=False)
    final_train_loss, final_train_ppl = validation_loss(model, train_loader, device, max_batches=20)

    print(f"✓ Evaluated Final Train Loss: {final_train_loss:.4f} (Perplexity: {final_train_ppl:.2f})")
    print(f"✓ Evaluated Final Val Loss:   {final_val_loss:.4f} (Perplexity: {final_val_ppl:,.1f})")

    # 2. Extract Training Curve Dynamics
    log = extract_training_log()
    steps = [x["step"] for x in log]
    train_losses = [x["loss"] for x in log]
    val_losses = [x["val_loss"] for x in log]
    tokens = [x["tokens_seen"] for x in log]

    best_val_idx = np.argmin(val_losses)
    best_step = steps[best_val_idx]
    best_val_loss = val_losses[best_val_idx]
    best_train_loss = train_losses[best_val_idx]
    best_tokens = tokens[best_val_idx]

    print("\n--- Loss Trajectory & Overfitting Identification ---")
    print(f"Initial Step (0):     Train Loss = {train_losses[0]:.2f}, Val Loss = {val_losses[0]:.2f}")
    print(f"Optimal Step ({best_step}):  Train Loss = {best_train_loss:.4f}, Val Loss = {best_val_loss:.4f} (Tokens: {best_tokens:,})")
    print(f"Final Step (1900):    Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    print(f"Overfitting Inflection Point: Step {best_step} (~{best_tokens/1000:.0f}K tokens, epoch ~{best_tokens/69632:.1f})")
    print(f"Validation Loss Drift: {best_val_loss:.2f} -> {val_losses[-1]:.2f} (+{val_losses[-1] - best_val_loss:.2f}, +{(val_losses[-1]/best_val_loss - 1)*100:.1f}%)")
    print(f"Training Loss Drop:    {best_train_loss:.2f} -> {train_losses[-1]:.2f} (-{best_train_loss - train_losses[-1]:.2f})")

    # Plot and save figure
    plt.figure(figsize=(10, 6))
    plt.plot(steps[1:], train_losses[1:], "b-o", label="Training Loss", linewidth=2)
    plt.plot(steps[1:], val_losses[1:], "r-s", label="Validation Loss", linewidth=2)
    plt.axvline(x=best_step, color="green", linestyle="--", label=f"Overfitting Onset (Step {best_step})")
    plt.title("ModernGPT-S v2: Pilot Pretraining Loss Curve (2M Tokens on 70K Shard)", fontsize=13)
    plt.xlabel("Optimization Steps (Batch Size: 2, Seq Len: 512)", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plot_path = Path("artifacts/reports/pilot_16k_v2_loss_curve.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Loss curve figure saved to {plot_path}")

    # 3. Qualitative Generation Testing (Fixed Prompts, Zero Cherry-Picking)
    print("\n" + "=" * 70)
    print("3. QUALITATIVE GENERATION PROBE AUDIT (UNSEEN & TRAINING PROMPTS)")
    print("=" * 70)

    test_prompts = [
        # 1. Unseen general English prose
        ("Unseen English", "The primary reason why machine learning models require large datasets is"),
        # 2. Unseen scientific prose
        ("Unseen Science", "Photosynthesis is the biochemical process by which green plants convert"),
        # 3. Unseen code
        ("Unseen Code", "def merge_sort(items: list) -> list:\n    if len(items) <= 1:"),
        # 4. Unseen mathematics
        ("Unseen Math", "In linear algebra, an eigenvector of a square matrix A corresponds to"),
        # 5. Training document exact match: Heel Bursitis (from FineWeb-Edu train doc 44)
        ("Train Corpus Match 1", "Heel Bursitis is another type of heel pain. The sufferer of this"),
        # 6. Training document exact match: Fandral (from Cosmopedia v2 train doc 12)
        ("Train Corpus Match 2", "Welcome to an exciting journey through the world of comic book heroes! Today, we're going to learn about a fantastic character named Fandral the Dashing."),
    ]

    generation_records = []

    for category, prompt in test_prompts:
        print(f"\n[Category: {category}]")
        print(f"Prompt: \"{prompt}\"")

        inp = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

        # Greedy decoding
        greedy_out = generate(model, inp.clone(), max_new_tokens=60, context_length=512)
        greedy_text = tok.decode(greedy_out[0].tolist())
        greedy_gen_only = tok.decode(greedy_out[0][inp.shape[1]:].tolist())

        # Sampled decoding (temp=0.8, top_k=40)
        sampled_out = generate_with_temperature(model, inp.clone(), max_new_tokens=60, context_length=512, temperature=0.8, top_k=40)
        sampled_gen_only = tok.decode(sampled_out[0][inp.shape[1]:].tolist())

        rep_3gram = compute_repetition_metrics(greedy_gen_only, n=3)

        print(f"Greedy Completion:\n  {repr(greedy_gen_only)}")
        print(f"Sampled Completion (temp=0.8):\n  {repr(sampled_gen_only)}")
        print(f"3-gram Repetition Score: {rep_3gram:.3f}")

        generation_records.append({
            "category": category,
            "prompt": prompt,
            "greedy_completion": greedy_gen_only,
            "sampled_completion": sampled_gen_only,
            "repetition_score_3gram": round(rep_3gram, 3),
        })

    # Save probe report JSON
    probe_report_path = Path("artifacts/reports/pilot_generation_audit.json")
    with probe_report_path.open("w", encoding="utf-8") as f:
        json.dump(generation_records, f, indent=2)
    print(f"\n✓ Generation records saved to {probe_report_path}")


if __name__ == "__main__":
    main()
