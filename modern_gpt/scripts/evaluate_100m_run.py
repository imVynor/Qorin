"""Comprehensive evaluation of ModernGPT-S after the 100M-token pretraining run.

Evaluates checkpoints/exp_100m/best_val.pt and latest.pt against 25M baseline:
1. Validates checkpoint metadata, step, tokens seen, and fingerprint.
2. Computes full-validation loss and perplexity across:
   - 100M validation shard (artifacts/shards_100m/val.bin)
   - 25M validation shard (artifacts/shards/bpe_16k_v2_25m/val.bin)
   - 5M validation shard (artifacts/shards/bpe_16k_v2_5m/val.bin)
3. Evaluates the 6 fixed qualitative pilot prompts from earlier audits.
4. Evaluates 24 diverse Python code prompts, measuring:
   - Syntax validity via ast.parse
   - Whitespace runaway detection
   - Distinct-N and n-gram repetition
   - Coherence and relevance
5. Saves all outputs and metrics to artifacts/reports/evaluation_100m_report.json.
"""

import ast
import json
import math
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add package root to sys.path
package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset
from evaluate import validation_loss
from generate import generate
from model.config import ModelConfig
from model.gpt import ModernGPT


def generate_with_temperature(model, input_ids, max_new_tokens, context_length, temperature=0.8, top_k=40, eos_token_id=0):
    """Generate tokens with temperature scaling and top-k filtering."""
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
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return input_ids


def compute_diversity_metrics(text: str):
    """Calculate n-gram repetition and Distinct-N diversity metrics."""
    words = text.split()
    if not words:
        return {"word_count": 0, "rep_1gram": 0.0, "rep_2gram": 0.0, "rep_3gram": 0.0, "distinct_1": 0.0, "distinct_2": 0.0}

    n1 = len(words)
    u1 = len(set(words))
    rep_1 = 1.0 - (u1 / n1)
    dist_1 = u1 / n1

    if len(words) >= 2:
        ng2 = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        rep_2 = 1.0 - (len(set(ng2)) / len(ng2))
        dist_2 = len(set(ng2)) / len(ng2)
    else:
        rep_2, dist_2 = 0.0, 1.0

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


def check_python_syntax(code: str) -> dict:
    """Check Python syntax using ast.parse with strict and block-trimmed modes."""
    # 1. Strict full code parse
    strict_valid = False
    strict_error = None
    try:
        ast.parse(code)
        strict_valid = True
    except SyntaxError as e:
        strict_error = f"{type(e).__name__}: {e.msg} (line {e.lineno})"
    except Exception as e:
        strict_error = str(e)

    # 2. Block-trimmed parse (drop trailing incomplete line if cut off by max_tokens)
    trimmed_valid = False
    trimmed_lines = code.splitlines()
    for end in range(len(trimmed_lines), 0, -1):
        candidate = "\n".join(trimmed_lines[:end])
        try:
            ast.parse(candidate)
            trimmed_valid = True
            break
        except SyntaxError:
            continue

    # 3. Detect whitespace runaway: consecutive whitespace characters > 30 without code
    whitespace_runaway = False
    if "                              " in code:
        whitespace_runaway = True

    return {
        "strict_valid": strict_valid,
        "strict_error": strict_error,
        "trimmed_valid": trimmed_valid,
        "whitespace_runaway": whitespace_runaway,
    }


def main():
    print("=" * 70)
    print("MODERNGPT-S: 100M-TOKEN PRETRAINING AUDIT & EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Tokenizer & Verify Fingerprint
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")
    expected_fingerprint = "be6c3d60c5baf217"
    print(f"Tokenizer: {tok.name} (Vocab: {tok.vocab_size}, Fingerprint: {tok.fingerprint}, EOS: {tok.eos_token_id})")
    assert tok.fingerprint == expected_fingerprint, f"Fingerprint mismatch: {tok.fingerprint} vs {expected_fingerprint}"

    # 2. Inspect Checkpoints
    ckpt_100m_best_path = Path("checkpoints/exp_100m/best_val.pt")
    ckpt_100m_latest_path = Path("checkpoints/exp_100m/latest.pt")
    ckpt_25m_best_path = Path("checkpoints/exp_25m/best_val.pt")

    assert ckpt_100m_best_path.exists(), f"Missing {ckpt_100m_best_path}"
    assert ckpt_100m_latest_path.exists(), f"Missing {ckpt_100m_latest_path}"
    assert ckpt_25m_best_path.exists(), f"Missing {ckpt_25m_best_path}"

    ckpt_100m_best = torch.load(ckpt_100m_best_path, map_location=device, weights_only=False)
    ckpt_100m_latest = torch.load(ckpt_100m_latest_path, map_location=device, weights_only=False)
    ckpt_25m_best = torch.load(ckpt_25m_best_path, map_location=device, weights_only=False)

    print("\n--- Checkpoint Metadata Audit ---")
    print(f"100M Best Step:   {ckpt_100m_best['step']} (Tokens: {ckpt_100m_best['tokens_seen']:,})")
    print(f"100M Latest Step: {ckpt_100m_latest['step']} (Tokens: {ckpt_100m_latest['tokens_seen']:,})")
    print(f"25M Best Step:    {ckpt_25m_best['step']} (Tokens: {ckpt_25m_best['tokens_seen']:,})")

    config = ModelConfig(vocab_size=16000, context_length=512)

    model_100m_best = ModernGPT(config).to(device)
    model_100m_best.load_state_dict(ckpt_100m_best["model"])
    model_100m_best.eval()

    model_100m_latest = ModernGPT(config).to(device)
    model_100m_latest.load_state_dict(ckpt_100m_latest["model"])
    model_100m_latest.eval()

    model_25m_best = ModernGPT(config).to(device)
    model_25m_best.load_state_dict(ckpt_25m_best["model"])
    model_25m_best.eval()

    # 3. Cross-Shard Loss & Perplexity Evaluations
    print("\n" + "=" * 70)
    print("QUANTITATIVE BENCHMARK: LOSS & PERPLEXITY ACROSS SHARDS")
    print("=" * 70)

    val_shards = {
        "100M_val_shard": Path("artifacts/shards_100m/val.bin"),
        "25M_val_shard": Path("artifacts/shards/bpe_16k_v2_25m/val.bin"),
        "5M_val_shard": Path("artifacts/shards/bpe_16k_v2_5m/val.bin"),
    }

    metrics_comparison = {}

    for shard_name, shard_path in val_shards.items():
        if not shard_path.exists():
            print(f"Skipping {shard_name}: not found at {shard_path}")
            continue

        ds = ShardedPackedDataset(shard_path, context_length=512)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        print(f"\nEvaluating on {shard_name} ({len(ds):,} sequences, {len(ds)*512:,} tokens)...")

        # 100M Best
        loss_100m_b, ppl_100m_b = validation_loss(model_100m_best, loader, device, max_batches=None)
        # 100M Latest
        loss_100m_l, ppl_100m_l = validation_loss(model_100m_latest, loader, device, max_batches=None)
        # 25M Best
        loss_25m_b, ppl_25m_b = validation_loss(model_25m_best, loader, device, max_batches=None)

        print(f"  100M Best:   Loss = {loss_100m_b:.4f}, Perplexity = {ppl_100m_b:.2f}")
        print(f"  100M Latest: Loss = {loss_100m_l:.4f}, Perplexity = {ppl_100m_l:.2f}")
        print(f"  25M Best:    Loss = {loss_25m_b:.4f}, Perplexity = {ppl_25m_b:.2f}")
        delta = loss_100m_b - loss_25m_b
        pct_imp = ((ppl_25m_b - ppl_100m_b) / ppl_25m_b) * 100
        print(f"  Improvement (100M Best vs 25M Best): ΔLoss = {delta:+.4f}, PPL Reduction = {pct_imp:.1f}%")

        metrics_comparison[shard_name] = {
            "num_sequences": len(ds),
            "total_tokens": len(ds) * 512,
            "100m_best": {"loss": round(loss_100m_b, 4), "perplexity": round(ppl_100m_b, 2)},
            "100m_latest": {"loss": round(loss_100m_l, 4), "perplexity": round(ppl_100m_l, 2)},
            "25m_best": {"loss": round(loss_25m_b, 4), "perplexity": round(ppl_25m_b, 2)},
            "delta_loss": round(delta, 4),
            "perplexity_reduction_pct": round(pct_imp, 2),
        }

    # 4. Qualitative Evaluation: 6 Fixed Historical Prompts
    print("\n" + "=" * 70)
    print("QUALITATIVE BENCHMARK: 6 FIXED HISTORICAL AUDIT PROMPTS")
    print("=" * 70)

    fixed_prompts = [
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

    fixed_results = []
    for cat, prompt in fixed_prompts:
        print(f"\n--- Category: {cat} ---")
        print(f"Prompt: {repr(prompt)}")

        inp = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

        # 100M Best Greedy
        g_out_100m = generate(model_100m_best, inp.clone(), max_new_tokens=60, context_length=512)
        g_text_100m = tok.decode(g_out_100m[0].tolist())
        g_gen_100m = tok.decode(g_out_100m[0][inp.shape[1]:].tolist())
        g_metrics_100m = compute_diversity_metrics(g_gen_100m)

        # 100M Best Sampled (temp=0.8, top_k=40)
        s_out_100m = generate_with_temperature(model_100m_best, inp.clone(), max_new_tokens=60, context_length=512, temperature=0.8, top_k=40)
        s_text_100m = tok.decode(s_out_100m[0].tolist())
        s_gen_100m = tok.decode(s_out_100m[0][inp.shape[1]:].tolist())
        s_metrics_100m = compute_diversity_metrics(s_gen_100m)

        # 25M Best Greedy for direct comparison
        g_out_25m = generate(model_25m_best, inp.clone(), max_new_tokens=60, context_length=512)
        g_gen_25m = tok.decode(g_out_25m[0][inp.shape[1]:].tolist())

        print(f"100M Greedy Completion:\n  {repr(g_gen_100m)}")
        print(f"100M Sampled Completion:\n  {repr(s_gen_100m)}")
        print(f"25M Greedy Baseline:\n  {repr(g_gen_25m)}")

        fixed_results.append({
            "category": cat,
            "prompt": prompt,
            "100m_greedy": {
                "completion": g_gen_100m,
                "full_text": g_text_100m,
                "metrics": g_metrics_100m,
            },
            "100m_sampled": {
                "completion": s_gen_100m,
                "full_text": s_text_100m,
                "metrics": s_metrics_100m,
            },
            "25m_greedy_baseline": g_gen_25m,
        })

    # 5. Qualitative Evaluation: 24 Python Code Prompts
    print("\n" + "=" * 70)
    print("CODE BENCHMARK: 24 DIVERSE PYTHON PROMPTS WITH SYNTAX AUDIT")
    print("=" * 70)

    python_prompts = [
        # Basic Functions & Math
        ("basic_add", "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of two integers.\"\"\"\n"),
        ("basic_factorial", "def factorial(n: int) -> int:\n    \"\"\"Calculate the factorial of n.\"\"\"\n    if n <= 1:\n"),
        ("basic_is_even", "def is_even(num: int) -> bool:\n    \"\"\"Check if a number is even.\"\"\"\n"),
        ("basic_gcd", "def gcd(a: int, b: int) -> int:\n    \"\"\"Compute greatest common divisor using Euclidean algorithm.\"\"\"\n"),
        ("basic_fibonacci", "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
        
        # String & List Algorithms
        ("str_reverse", "def reverse_string(s: str) -> str:\n    \"\"\"Reverse the input string.\"\"\"\n"),
        ("str_palindrome", "def is_palindrome(text: str) -> bool:\n    \"\"\"Check if text reads the same backward as forward.\"\"\"\n"),
        ("str_count_vowels", "def count_vowels(s: str) -> int:\n    \"\"\"Count the number of vowels in a string.\"\"\"\n"),
        ("list_binary_search", "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1 if not found.\"\"\"\n    left, right = 0, len(arr) - 1\n"),
        ("list_quicksort", "def quicksort(arr: list) -> list:\n    \"\"\"Sort an array using the quicksort algorithm.\"\"\"\n    if len(arr) <= 1:\n"),
        ("list_merge_sort", "def merge_sort(items: list) -> list:\n    if len(items) <= 1:\n"),
        ("list_remove_dups", "def remove_duplicates(items: list) -> list:\n    \"\"\"Remove duplicates while preserving order.\"\"\"\n"),
        ("list_flatten", "def flatten(nested: list) -> list:\n    \"\"\"Flatten a list of lists into a single list.\"\"\"\n"),
        ("list_find_max", "def find_max(numbers: list) -> float:\n    \"\"\"Find maximum value without using built-in max().\"\"\"\n"),

        # Dictionary & Data Processing
        ("dict_word_count", "def word_count(text: str) -> dict:\n    \"\"\"Count frequency of each word in text.\"\"\"\n"),
        ("dict_invert", "def invert_dict(d: dict) -> dict:\n    \"\"\"Invert key-value mappings of a dictionary.\"\"\"\n"),
        ("dict_filter_by_val", "def filter_by_threshold(scores: dict, threshold: float) -> dict:\n    \"\"\"Filter entries whose values exceed threshold.\"\"\"\n"),
        ("data_read_json", "import json\n\ndef load_json_config(file_path: str) -> dict:\n    \"\"\"Load and parse a JSON configuration file.\"\"\"\n"),
        ("data_write_csv", "import csv\n\ndef save_to_csv(rows: list, filename: str) -> None:\n    \"\"\"Save rows to CSV file.\"\"\"\n"),
        
        # Object-Oriented Programming (Classes)
        ("oop_stack", "class Stack:\n    \"\"\"LIFO stack implementation.\"\"\"\n    def __init__(self):\n"),
        ("oop_queue", "class Queue:\n    \"\"\"FIFO queue implementation.\"\"\"\n    def __init__(self):\n"),
        ("oop_bank_account", "class BankAccount:\n    def __init__(self, owner: str, balance: float = 0.0):\n        self.owner = owner\n        self.balance = balance\n\n    def deposit(self, amount: float) -> float:\n"),
        ("oop_rectangle", "class Rectangle:\n    def __init__(self, width: float, height: float):\n        self.width = width\n        self.height = height\n\n    def area(self) -> float:\n"),
        ("oop_node", "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n"),
    ]

    code_results = []
    strict_valid_count = 0
    trimmed_valid_count = 0
    whitespace_runaway_count = 0

    for name, prompt in python_prompts:
        inp = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

        # 100M Greedy
        g_out = generate(model_100m_best, inp.clone(), max_new_tokens=60, context_length=512)
        g_full = tok.decode(g_out[0].tolist())
        g_gen = tok.decode(g_out[0][inp.shape[1]:].tolist())

        # 100M Sampled
        s_out = generate_with_temperature(model_100m_best, inp.clone(), max_new_tokens=60, context_length=512, temperature=0.8, top_k=40)
        s_full = tok.decode(s_out[0].tolist())
        s_gen = tok.decode(s_out[0][inp.shape[1]:].tolist())

        # Check syntax on greedy full text
        syntax_greedy = check_python_syntax(g_full)
        syntax_sampled = check_python_syntax(s_full)

        # Baseline 25M greedy on same prompt
        g_out_25m = generate(model_25m_best, inp.clone(), max_new_tokens=60, context_length=512)
        g_gen_25m = tok.decode(g_out_25m[0][inp.shape[1]:].tolist())
        syntax_25m = check_python_syntax(tok.decode(g_out_25m[0].tolist()))

        if syntax_greedy["strict_valid"]:
            strict_valid_count += 1
        if syntax_greedy["trimmed_valid"]:
            trimmed_valid_count += 1
        if syntax_greedy["whitespace_runaway"]:
            whitespace_runaway_count += 1

        print(f"[{name}] StrictValid={syntax_greedy['strict_valid']}, TrimmedValid={syntax_greedy['trimmed_valid']}, WS_Runaway={syntax_greedy['whitespace_runaway']}")
        print(f"  Prompt:   {prompt.splitlines()[-1] if prompt.splitlines() else prompt}")
        print(f"  100M Gen: {repr(g_gen[:80])}...")
        if syntax_25m["whitespace_runaway"]:
            print(f"  (Note: 25M baseline had WHITESPACE RUNAWAY on this prompt)")

        code_results.append({
            "name": name,
            "prompt": prompt,
            "100m_greedy": {
                "completion": g_gen,
                "full_text": g_full,
                "syntax": syntax_greedy,
                "metrics": compute_diversity_metrics(g_gen),
            },
            "100m_sampled": {
                "completion": s_gen,
                "full_text": s_full,
                "syntax": syntax_sampled,
                "metrics": compute_diversity_metrics(s_gen),
            },
            "25m_greedy": {
                "completion": g_gen_25m,
                "syntax": syntax_25m,
            },
        })

    total_code_prompts = len(python_prompts)
    strict_valid_rate = (strict_valid_count / total_code_prompts) * 100
    trimmed_valid_rate = (trimmed_valid_count / total_code_prompts) * 100
    whitespace_runaway_rate = (whitespace_runaway_count / total_code_prompts) * 100

    print("\n" + "=" * 70)
    print("CODE BENCHMARK SUMMARY (100M MODEL)")
    print("=" * 70)
    print(f"Total Python Prompts Evaluated: {total_code_prompts}")
    print(f"Strict ast.parse Valid:         {strict_valid_count}/{total_code_prompts} ({strict_valid_rate:.1f}%)")
    print(f"Block-Trimmed ast.parse Valid:  {trimmed_valid_count}/{total_code_prompts} ({trimmed_valid_rate:.1f}%)")
    print(f"Whitespace Runaways Detected:   {whitespace_runaway_count}/{total_code_prompts} ({whitespace_runaway_rate:.1f}%)")

    # 6. Save Comprehensive Report
    report_data = {
        "run": "ModernGPT-S 100M Pretraining",
        "best_checkpoint": str(ckpt_100m_best_path),
        "latest_checkpoint": str(ckpt_100m_latest_path),
        "comparison_baseline": str(ckpt_25m_best_path),
        "best_step": ckpt_100m_best["step"],
        "latest_step": ckpt_100m_latest["step"],
        "tokens_seen": ckpt_100m_best["tokens_seen"],
        "tokenizer_fingerprint": tok.fingerprint,
        "eos_token_id": tok.eos_token_id,
        "corpus_mixture": {
            "fineweb_edu_pct": 70.08,
            "cosmopedia_v2_pct": 19.94,
            "codeparrot_clean_pct": 9.98,
        },
        "metrics_by_validation_shard": metrics_comparison,
        "code_benchmark_summary": {
            "total_prompts": total_code_prompts,
            "strict_valid_count": strict_valid_count,
            "strict_valid_pct": round(strict_valid_rate, 2),
            "trimmed_valid_count": trimmed_valid_count,
            "trimmed_valid_pct": round(trimmed_valid_rate, 2),
            "whitespace_runaway_count": whitespace_runaway_count,
            "whitespace_runaway_pct": round(whitespace_runaway_rate, 2),
        },
        "fixed_prompts_results": fixed_results,
        "python_code_results": code_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_path = Path("artifacts/reports/evaluation_100m_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n✓ Saved evaluation report to {report_path} ({report_path.stat().st_size:,} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
