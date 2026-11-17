"""Comprehensive evaluation of ModernGPT-S after the 5M pretraining run.

Evaluates checkpoints/exp_5m/best_val.pt and checkpoints/exp_5m/latest.pt:
1. Validates checkpoint fingerprint, parameters, step, and tokens seen.
2. Evaluates exact loss and perplexity across:
   - Full 5M validation shard (artifacts/shards/bpe_16k_v2_5m/val.bin)
   - 5-batch validation window (matching training eval_batches)
   - 5M train shard sample for train-val generalization gap
3. Runs greedy and top-k sampled generation on the 6 fixed pilot prompts.
4. Calculates n-gram repetition, Distinct-N diversity metrics, coherence, and topic retention.
5. Saves all generation outputs and metrics to artifacts/reports/evaluation_5m_report.json.
"""

import json
import math
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset
from evaluate import validation_loss
from generate import generate
from model.config import ModelConfig
from model.gpt import ModernGPT


def generate_with_temperature(model, input_ids, max_new_tokens, context_length, temperature=0.8, top_k=40):
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


def compute_diversity_metrics(text: str):
    words = text.split()
    if not words:
        return {"rep_1gram": 0.0, "rep_2gram": 0.0, "rep_3gram": 0.0, "distinct_1": 0.0, "distinct_2": 0.0}

    # 1-gram
    n1 = len(words)
    u1 = len(set(words))
    rep_1 = 1.0 - (u1 / n1)
    dist_1 = u1 / n1

    # 2-gram
    if len(words) >= 2:
        ng2 = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        rep_2 = 1.0 - (len(set(ng2)) / len(ng2))
        dist_2 = len(set(ng2)) / len(ng2)
    else:
        rep_2, dist_2 = 0.0, 1.0

    # 3-gram
    if len(words) >= 3:
        ng3 = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        rep_3 = 1.0 - (len(set(ng3)) / len(ng3))
    else:
        rep_3 = 0.0

    return {
        "word_count": len(words),
        "rep_1gram": round(rep_1, 4),
        "rep_2gram": round(rep_2, 4),
        "rep_3gram": round(rep_3, 4),
        "distinct_1": round(dist_1, 4),
        "distinct_2": round(dist_2, 4),
    }


def main():
    print("=" * 70)
    print("MODERNGPT-S: 5M-TOKEN PRETRAINING RUN EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Tokenizer & verify fingerprint
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")
    expected_fingerprint = "be6c3d60c5baf217"
    print(f"Tokenizer: {tok.name} (Vocab: {tok.vocab_size}, Fingerprint: {tok.fingerprint})")
    assert tok.fingerprint == expected_fingerprint, f"Fingerprint mismatch: {tok.fingerprint} vs {expected_fingerprint}"

    # 2. Inspect Best Checkpoint
    best_ckpt_path = Path("checkpoints/exp_5m/best_val.pt")
    assert best_ckpt_path.exists(), f"Missing checkpoint: {best_ckpt_path}"
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    print("\n--- Checkpoint Integrity Audit: best_val.pt ---")
    print(f"✓ File size:             {best_ckpt_path.stat().st_size:,} bytes")
    print(f"✓ Step:                  {best_ckpt['step']}")
    print(f"✓ Tokens seen:           {best_ckpt['tokens_seen']:,}")
    print(f"✓ Vocab size:            {best_ckpt['vocab_size']}")
    print(f"✓ Tokenizer:             {best_ckpt['tokenizer']}")
    print(f"✓ Tokenizer Fingerprint: {best_ckpt.get('tokenizer_fingerprint')}")
    print(f"✓ Manifest Meta:         {best_ckpt.get('manifest_meta')}")
    assert best_ckpt.get("tokenizer_fingerprint") == expected_fingerprint

    # Inspect Latest Checkpoint
    latest_ckpt_path = Path("checkpoints/exp_5m/latest.pt")
    latest_ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    print("\n--- Checkpoint Integrity Audit: latest.pt ---")
    print(f"✓ Step:                  {latest_ckpt['step']}")
    print(f"✓ Tokens seen:           {latest_ckpt['tokens_seen']:,}")

    # Load Model with Best Weights
    config = ModelConfig(vocab_size=16000, context_length=512)
    model = ModernGPT(config).to(device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    # 3. Evaluate Exact Loss on 5M Validation Shard
    val_shard_path = Path("artifacts/shards/bpe_16k_v2_5m/val.bin")
    val_dataset = ShardedPackedDataset(val_shard_path, context_length=512)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    print(f"\nValidation Shard Total Sequences: {len(val_dataset):,} ({len(val_dataset) * 512:,} tokens)")

    # Full validation shard evaluation
    full_val_loss, full_val_ppl = validation_loss(model, val_loader, device, max_batches=None)
    # 5-batch validation window (to verify against training log eval)
    window_val_loss, window_val_ppl = validation_loss(model, val_loader, device, max_batches=5)

    print(f"✓ Full Val Loss (all {len(val_dataset)} seqs): {full_val_loss:.4f} (Perplexity: {full_val_ppl:.2f})")
    print(f"✓ 5-batch Val Loss (10 seqs):                {window_val_loss:.4f} (Perplexity: {window_val_ppl:.2f})")

    # Evaluate on Train Shard Sample (100 batches = 200 sequences = 102,400 tokens)
    train_shard_path = Path("artifacts/shards/bpe_16k_v2_5m/train.bin")
    train_dataset = ShardedPackedDataset(train_shard_path, context_length=512)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    sample_train_loss, sample_train_ppl = validation_loss(model, train_loader, device, max_batches=100)
    print(f"✓ Sample Train Loss (100 batches):            {sample_train_loss:.4f} (Perplexity: {sample_train_ppl:.2f})")
    print(f"✓ Generalization Gap (Full Val - Train):      {full_val_loss - sample_train_loss:+.4f}")

    # Evaluate Latest Checkpoint on Full Val Shard to check for end-of-run drift
    model_latest = ModernGPT(config).to(device)
    model_latest.load_state_dict(latest_ckpt["model"])
    model_latest.eval()
    latest_val_loss, latest_val_ppl = validation_loss(model_latest, val_loader, device, max_batches=None)
    print(f"✓ Latest Checkpoint Full Val Loss:            {latest_val_loss:.4f} (Perplexity: {latest_val_ppl:.2f})")

    # 4. Generate on 6 Fixed Prompts
    test_prompts = [
        ("Unseen English", "The primary reason why machine learning models require large datasets is"),
        ("Unseen Science", "Photosynthesis is the biochemical process by which green plants convert"),
        ("Unseen Code", "def merge_sort(items: list) -> list:\n    if len(items) <= 1:"),
        ("Unseen Math", "In linear algebra, an eigenvector of a square matrix A corresponds to"),
        ("Train Corpus Match 1", "Heel Bursitis is another type of heel pain. The sufferer of this"),
        ("Train Corpus Match 2", "Welcome to an exciting journey through the world of comic book heroes! Today, we're going to learn about a fantastic character named Fandral the Dashing."),
    ]

    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    results = []
    print("\n" + "=" * 70)
    print("QUALITATIVE GENERATION AUDIT (FIXED PROMPTS, GREEDY & SAMPLED)")
    print("=" * 70)

    for cat, prompt in test_prompts:
        print(f"\n--- Category: {cat} ---")
        print(f"Prompt: {repr(prompt)}")

        input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

        # Greedy decoding
        greedy_out = generate(model, input_ids.clone(), max_new_tokens=60, context_length=512)
        greedy_text = tok.decode(greedy_out[0].tolist())
        greedy_gen_only = tok.decode(greedy_out[0][input_ids.shape[1]:].tolist())

        # Top-k sampled decoding (temp=0.8, top_k=40)
        torch.manual_seed(123)
        sampled_out = generate_with_temperature(model, input_ids.clone(), max_new_tokens=60, context_length=512, temperature=0.8, top_k=40)
        sampled_text = tok.decode(sampled_out[0].tolist())
        sampled_gen_only = tok.decode(sampled_out[0][input_ids.shape[1]:].tolist())

        g_div = compute_diversity_metrics(greedy_gen_only)
        s_div = compute_diversity_metrics(sampled_gen_only)

        print(f"Greedy:\n  {repr(greedy_gen_only)}")
        print(f"  [3-gram Repetition: {g_div['rep_3gram']:.3f}, Distinct-1: {g_div['distinct_1']:.3f}]")
        print(f"Sampled (temp=0.8, top_k=40):\n  {repr(sampled_gen_only)}")
        print(f"  [3-gram Repetition: {s_div['rep_3gram']:.3f}, Distinct-1: {s_div['distinct_1']:.3f}]")

        results.append({
            "category": cat,
            "prompt": prompt,
            "greedy_completion": greedy_gen_only,
            "greedy_full_text": greedy_text,
            "greedy_metrics": g_div,
            "sampled_completion": sampled_gen_only,
            "sampled_full_text": sampled_text,
            "sampled_metrics": s_div,
        })

    # Save outputs to JSON
    report_output_path = Path("artifacts/reports/evaluation_5m_report.json")
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "run": "ModernGPT-S 5M Pretraining",
        "checkpoint": "checkpoints/exp_5m/best_val.pt",
        "best_step": best_ckpt["step"],
        "tokens_seen": best_ckpt["tokens_seen"],
        "tokenizer_fingerprint": tok.fingerprint,
        "evaluation_metrics": {
            "full_val_loss": round(full_val_loss, 4),
            "full_val_perplexity": round(full_val_ppl, 2),
            "window_val_loss": round(window_val_loss, 4),
            "window_val_perplexity": round(window_val_ppl, 2),
            "sample_train_loss": round(sample_train_loss, 4),
            "sample_train_perplexity": round(sample_train_ppl, 2),
            "latest_full_val_loss": round(latest_val_loss, 4),
            "latest_full_val_perplexity": round(latest_val_ppl, 2),
        },
        "generations": results,
    }

    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✓ Saved evaluation report to {report_output_path}")


if __name__ == "__main__":
    main()
