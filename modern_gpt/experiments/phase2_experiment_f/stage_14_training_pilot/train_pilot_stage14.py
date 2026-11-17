"""
QORIN Experiment F — Stage 14: Controlled Multi-Turn Fine-Tuning Pilot
Executes authorized 100 optimizer updates on Stage 13 dataset.
Parent: Experiment D Step 425 (ckpt_experiment_d_best.pt)
"""

import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import yaml

REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig

# Paths and Expected Hashes
PARENT_CKPT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
EXPECTED_PARENT_SHA = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXPECTED_TOK_SHA = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

DATA_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_13_dataset"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "validation.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

PILOT_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot"
CKPT_DIR = PILOT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_telemetry() -> Dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "temp_c": float(parts[0]),
            "power_w": float(parts[1]),
            "utilization_pct": float(parts[2]),
            "memory_used_mib": float(parts[3]),
            "memory_total_mib": float(parts[4]),
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e), "temp_c": -1, "memory_used_mib": -1, "timestamp": time.time()}


def get_lr(step: int, total_steps: int = 100, warmup_steps: int = 20, peak_lr: float = 1.5e-5, min_lr: float = 1.5e-6) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class MultiTurnDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tok: Tokenizer):
        self.samples = []
        for rec in records:
            chatml = rec["serialized_chatml"]
            loss_mask = rec["loss_mask"]
            enc = tok.encode(chatml)
            input_ids = enc.ids
            assert len(input_ids) == len(loss_mask), f"Length mismatch: {len(input_ids)} vs {len(loss_mask)}"
            labels = [input_ids[t] if loss_mask[t] == 1 else -100 for t in range(len(input_ids))]

            self.samples.append({
                "id": rec["id"],
                "corpus": rec["corpus"],
                "category": rec["category"],
                "domain": rec.get("domain", ""),
                "prompt": rec["prompt"],
                "target_response": rec.get("target_response", ""),
                "serialized_chatml": chatml,
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                "total_tokens": len(input_ids),
                "assistant_tokens": sum(loss_mask)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = batch[0]
    return {
        "id": b["id"],
        "corpus": b["corpus"],
        "category": b["category"],
        "domain": b["domain"],
        "prompt": b["prompt"],
        "target_response": b["target_response"],
        "serialized_chatml": b["serialized_chatml"],
        "input_ids": torch.tensor([b["input_ids"]], dtype=torch.long),
        "labels": torch.tensor([b["labels"]], dtype=torch.long),
        "total_tokens": b["total_tokens"],
        "assistant_tokens": b["assistant_tokens"]
    }


@torch.inference_mode()
def generate_greedy(model, tok, prompt_text: str, max_new_tokens: int = 160, device: str = "cuda") -> str:
    input_ids = tok.encode(prompt_text).ids
    generated = list(input_ids)
    stop_ids = {5, 9}  # 5: <|im_end|>, 9: <|endoftext|>

    for _ in range(max_new_tokens):
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inp)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if next_id in stop_ids:
            break
        generated.append(next_id)

    new_ids = generated[len(input_ids):]
    return tok.decode(new_ids, skip_special_tokens=False)


def compute_validation_loss(model, val_dataset: MultiTurnDataset, device: str = "cuda") -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    corpus_loss = {}
    corpus_tokens = {}
    category_loss = {}
    category_tokens = {}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for sample in val_dataset:
            inp = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
            lab = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
            corp = sample["corpus"]
            cat = sample["category"]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = lab[..., 1:].contiguous()

                mask = (shift_labels != -100)
                n_tok = mask.sum().item()
                if n_tok == 0:
                    continue

                sample_loss = loss_fn(shift_logits[mask], shift_labels[mask]).item()
                total_loss += sample_loss
                total_target_tokens += n_tok

                corpus_loss[corp] = corpus_loss.get(corp, 0.0) + sample_loss
                corpus_tokens[corp] = corpus_tokens.get(corp, 0.0) + n_tok

                category_loss[cat] = category_loss.get(cat, 0.0) + sample_loss
                category_tokens[cat] = category_tokens.get(cat, 0.0) + n_tok

    overall_val_loss = total_loss / max(1, total_target_tokens)
    ppl = math.exp(min(20.0, overall_val_loss))

    per_corpus = {k: round(corpus_loss[k] / max(1, corpus_tokens[k]), 4) for k in corpus_loss}
    per_cat = {k: round(category_loss[k] / max(1, category_tokens[k]), 4) for k in category_loss}

    return {
        "val_loss": round(overall_val_loss, 4),
        "perplexity": round(ppl, 4),
        "total_target_tokens": total_target_tokens,
        "per_corpus_loss": per_corpus,
        "per_category_loss": per_cat
    }


def evaluate_autonomous_capabilities(model, tok, val_records: List[Dict[str, Any]], device: str = "cuda") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()

    # System prompt for prompt construction
    sys_prompt = "You are QORIN, a calibrated, evidence-grounded AI assistant.\nYou have access to the following tools:\n- `web_search`: Searches verified databases for factual information.\n- `python_interpreter`: Executes isolated Python scripts for calculation and data processing.\n\nUse tools only when strictly necessary. When direct knowledge suffices, answer directly. Always ground answers on retrieved evidence with [1], [2] citations."

    # 1. Multi-turn evaluation (40 records)
    mturn_records = [r for r in val_records if r["corpus"] == "corpus_multiturn"]
    mturn_results = []
    
    first_turn_web_success = 0
    web_arg_valid = 0
    second_turn_python_success = 0
    python_arg_valid = 0
    id_handling_success = 0
    final_synthesis_success = 0
    full_trajectory_success = 0

    std_mturn_count = sum(1 for r in mturn_records if r["category"] == "MULTITURN_STANDARD_WEB_PYTHON")

    for r in mturn_records:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        target = r.get("target_response", "")

        # Parse ground-truth turns from ChatML to extract ground-truth observations
        turns = re.findall(r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>', r["serialized_chatml"], re.DOTALL)
        obs_1 = ""
        obs_2 = ""
        for h, c in turns:
            h_str = h.strip()
            if "tool" in h_str and "call_001" in h_str:
                obs_1 = c.strip()
            elif "tool" in h_str and "call_002" in h_str:
                obs_2 = c.strip()

        # Turn 1: Model generates first tool call
        p1 = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        gen_1 = generate_greedy(model, tok, p1, max_new_tokens=100, device=device)

        has_web = "web_search" in gen_1
        has_id_1 = "call_001" in gen_1
        valid_query = False
        try:
            m = re.search(r'\{.*\}', gen_1, re.DOTALL)
            if m:
                args = json.loads(m.group(0))
                if "query" in args and len(args["query"]) > 2:
                    valid_query = True
        except Exception:
            valid_query = False

        if has_web:
            first_turn_web_success += 1
        if valid_query:
            web_arg_valid += 1
        if has_id_1:
            id_handling_success += 1

        # Turn 2: Provide observation 1
        gen_2 = ""
        has_python = False
        valid_code = False
        has_id_2 = False

        if cat == "MULTITURN_STANDARD_WEB_PYTHON":
            p2 = f"{p1}{gen_1}<|im_end|>\n<|im_start|>tool name=web_search id=call_001\n{obs_1}<|im_end|>\n<|im_start|>assistant\n"
            gen_2 = generate_greedy(model, tok, p2, max_new_tokens=120, device=device)

            has_python = "python_interpreter" in gen_2
            has_id_2 = "call_002" in gen_2
            try:
                m2 = re.search(r'\{.*\}', gen_2, re.DOTALL)
                if m2:
                    args2 = json.loads(m2.group(0))
                    if "code" in args2:
                        ast.parse(args2["code"])
                        valid_code = True
            except Exception:
                valid_code = False

            if has_python:
                second_turn_python_success += 1
            if valid_code:
                python_arg_valid += 1
            if has_id_2:
                id_handling_success += 1

            # Turn 3: Provide observation 2
            p3 = f"{p2}{gen_2}<|im_end|>\n<|im_start|>tool name=python_interpreter id=call_002\n{obs_2}<|im_end|>\n<|im_start|>assistant\n"
            gen_3 = generate_greedy(model, tok, p3, max_new_tokens=120, device=device)

            has_cite = "[1]" in gen_3 or "[2]" in gen_3
            has_substance = len(gen_3.strip()) > 30
            if has_cite and has_substance:
                final_synthesis_success += 1

            trajectory_ok = has_web and valid_query and has_python and valid_code and has_cite
            if trajectory_ok:
                full_trajectory_success += 1

            mturn_results.append({
                "id": rid,
                "category": cat,
                "turn1_gen": gen_1,
                "turn2_gen": gen_2,
                "turn3_gen": gen_3,
                "turn1_web": has_web,
                "turn2_python": has_python,
                "trajectory_success": trajectory_ok
            })
        else:
            # Other multi-turn categories (sufficient direct, failure recovery, contradictory)
            p2 = f"{p1}{gen_1}<|im_end|>\n<|im_start|>tool name=web_search id=call_001\n{obs_1}<|im_end|>\n<|im_start|>assistant\n"
            gen_2 = generate_greedy(model, tok, p2, max_new_tokens=120, device=device)
            mturn_results.append({
                "id": rid,
                "category": cat,
                "turn1_gen": gen_1,
                "turn2_gen": gen_2,
                "turn1_web": has_web,
                "turn2_python": False,
                "trajectory_success": has_web and len(gen_2.strip()) > 20
            })

    multiturn_metrics = {
        "first_turn_web_accuracy": round(first_turn_web_success / max(1, len(mturn_records)), 4),
        "web_argument_validity": round(web_arg_valid / max(1, len(mturn_records)), 4),
        "second_turn_python_accuracy": round(second_turn_python_success / max(1, std_mturn_count), 4),
        "python_argument_validity": round(python_arg_valid / max(1, std_mturn_count), 4),
        "id_handling_success_rate": round(id_handling_success / max(1, len(mturn_records) + std_mturn_count), 4),
        "final_synthesis_accuracy": round(final_synthesis_success / max(1, std_mturn_count), 4),
        "full_trajectory_success_rate": round(full_trajectory_success / max(1, std_mturn_count), 4),
        "full_trajectory_successes": full_trajectory_success,
        "standard_multiturn_total": std_mturn_count
    }

    # 2. Decision calibration evaluation (30 records)
    cal_records = [r for r in val_records if r["corpus"] == "corpus_calibration"]
    direct_concept_fp = 0
    direct_concept_total = 0
    provided_calc_tp = 0
    provided_calc_total = 0
    factual_web_tp = 0
    factual_web_total = 0
    contrastive_correct = 0
    contrastive_total = 0
    total_tool_calls_emitted = 0
    total_tool_calls_needed = 0
    correct_tool_calls = 0

    cal_results = []
    for r in cal_records:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]

        p = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        gen = generate_greedy(model, tok, p, max_new_tokens=90, device=device)
        has_tool = "<tool_call" in gen or "{\"name\":" in gen

        if has_tool:
            total_tool_calls_emitted += 1

        correct = False
        if cat == "CAL_DIRECT_CONCEPT":
            direct_concept_total += 1
            if has_tool:
                direct_concept_fp += 1
            else:
                correct = True
        elif cat == "CAL_DIRECT_PROVIDED_CALC":
            provided_calc_total += 1
            total_tool_calls_needed += 1
            if "python_interpreter" in gen:
                provided_calc_tp += 1
                correct_tool_calls += 1
                correct = True
        elif cat == "CAL_DIRECT_FACTUAL_WEB":
            factual_web_total += 1
            total_tool_calls_needed += 1
            if "web_search" in gen:
                factual_web_tp += 1
                correct_tool_calls += 1
                correct = True
        elif cat == "CAL_CONTRASTIVE_PAIRS":
            contrastive_total += 1
            # Check target intent
            if "python_interpreter" in r["serialized_chatml"]:
                total_tool_calls_needed += 1
                if "python_interpreter" in gen:
                    contrastive_correct += 1
                    correct_tool_calls += 1
                    correct = True
            elif "web_search" in r["serialized_chatml"]:
                total_tool_calls_needed += 1
                if "web_search" in gen:
                    contrastive_correct += 1
                    correct_tool_calls += 1
                    correct = True
            else:
                if not has_tool:
                    contrastive_correct += 1
                    correct = True

        cal_results.append({
            "id": rid,
            "category": cat,
            "generated": gen[:100],
            "correct": correct
        })

    intent_precision = round(correct_tool_calls / max(1, total_tool_calls_emitted), 4) if total_tool_calls_emitted > 0 else 1.0
    intent_recall = round(correct_tool_calls / max(1, total_tool_calls_needed), 4)
    intent_f1 = round(2 * intent_precision * intent_recall / max(1e-6, intent_precision + intent_recall), 4)

    cal_metrics = {
        "intent_precision": intent_precision,
        "intent_recall": intent_recall,
        "intent_f1": intent_f1,
        "direct_concept_fpr": round(direct_concept_fp / max(1, direct_concept_total), 4),
        "unnecessary_tool_calls": direct_concept_fp,
        "missed_tool_calls": total_tool_calls_needed - correct_tool_calls,
        "python_selection_recall": round(provided_calc_tp / max(1, provided_calc_total), 4),
        "web_selection_recall": round(factual_web_tp / max(1, factual_web_total), 4),
        "contrastive_accuracy": round(contrastive_correct / max(1, contrastive_total), 4),
        "cal_results": cal_results
    }

    # 3. Evidence synthesis evaluation (30 records)
    syn_records = [r for r in val_records if r["corpus"] == "corpus_synthesis"]
    supported_claims_correct = 0
    citation_correct = 0
    partial_support_correct = 0
    contradiction_detected = 0
    insufficient_acknowledged = 0

    full_count = sum(1 for r in syn_records if r["category"] == "SYN_FULL_SUPPORT")
    part_count = sum(1 for r in syn_records if r["category"] == "SYN_PARTIAL_SUPPORT")
    contra_count = sum(1 for r in syn_records if r["category"] == "SYN_CONTRADICTORY_EVIDENCE")
    insuff_count = sum(1 for r in syn_records if r["category"] == "SYN_INSUFFICIENT_EVIDENCE")

    syn_results = []
    for r in syn_records:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]

        # Extract tool observation from chatml
        turns = re.findall(r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>', r["serialized_chatml"], re.DOTALL)
        tool_obs = ""
        for h, c in turns:
            if "tool" in h.strip():
                tool_obs = c.strip()

        p = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>tool\n{tool_obs}<|im_end|>\n<|im_start|>assistant\n"
        gen = generate_greedy(model, tok, p, max_new_tokens=100, device=device)

        gen_lower = gen.lower()
        has_cite = "[1]" in gen or "[2]" in gen

        if cat == "SYN_FULL_SUPPORT":
            if has_cite:
                citation_correct += 1
            if len(gen.strip()) > 30 and ("confirm" in gen_lower or "report" in gen_lower or "indicate" in gen_lower or "record" in gen_lower or has_cite):
                supported_claims_correct += 1
        elif cat == "SYN_PARTIAL_SUPPORT":
            if "partial" in gen_lower or "approximate" in gen_lower or "estimated" in gen_lower or "preliminary" in gen_lower or "however" in gen_lower or "while" in gen_lower:
                partial_support_correct += 1
        elif cat == "SYN_CONTRADICTORY_EVIDENCE":
            if "conflict" in gen_lower or "discrepanc" in gen_lower or "differ" in gen_lower or "vary" in gen_lower or "range" in gen_lower or "between" in gen_lower:
                contradiction_detected += 1
        elif cat == "SYN_INSUFFICIENT_EVIDENCE":
            if "insufficient" in gen_lower or "not provided" in gen_lower or "does not contain" in gen_lower or "no record" in gen_lower or "unknown" in gen_lower:
                insufficient_acknowledged += 1

        syn_results.append({
            "id": rid,
            "category": cat,
            "generated": gen[:100],
            "has_citation": has_cite
        })

    syn_metrics = {
        "strict_supported_precision": round(supported_claims_correct / max(1, full_count), 4),
        "citation_correctness": round(citation_correct / max(1, full_count), 4),
        "partial_support_rate": round(partial_support_correct / max(1, part_count), 4),
        "contradiction_rate": round(contradiction_detected / max(1, contra_count), 4),
        "insufficient_evidence_rate": round(insufficient_acknowledged / max(1, insuff_count), 4),
        "unsupported_claim_rate": round(1.0 - (supported_claims_correct / max(1, full_count)), 4),
        "syn_results": syn_results
    }

    model.train()
    return multiturn_metrics, cal_metrics, syn_metrics, mturn_results


def run_training_pilot():
    print("=" * 70)
    print("QORIN EXPERIMENT F — STAGE 14: CONTROLLED MULTI-TURN FINE-TUNING PILOT")
    print("=" * 70)

    # 1. Verification of Immutable Parent Checkpoint & Tokenizer
    print("\n[Phase 1] Checkpoint & Integrity Verification...")
    assert sha256_file(PARENT_CKPT_PATH) == EXPECTED_PARENT_SHA, "Parent Checkpoint SHA mismatch!"
    print(f"  Parent Checkpoint verified: {PARENT_CKPT_PATH.name} (SHA: {EXPECTED_PARENT_SHA[:16]}...)")

    assert sha256_file(TOK_PATH) == EXPECTED_TOK_SHA, "Tokenizer SHA mismatch!"
    print(f"  Tokenizer verified: {TOK_PATH.name} (SHA: {EXPECTED_TOK_SHA[:16]}...)")

    tok = Tokenizer.from_file(str(TOK_PATH))

    # 2. Dataset Loading & Stratified Sampler
    print("\n[Phase 2] Loading Stage 13 Datasets...")
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]

    print(f"  Train: {len(train_records)} records | Val: {len(val_records)} records | Test: {len(test_records)} records (held out)")

    # Prepare deterministic balanced sequence: 40% A, 30% B, 30% C
    a_pool = [r for r in train_records if r["corpus"] == "corpus_multiturn"]
    b_pool = [r for r in train_records if r["corpus"] == "corpus_calibration"]
    c_pool = [r for r in train_records if r["corpus"] == "corpus_synthesis"]

    rng = random.Random(42)
    rng.shuffle(a_pool)
    rng.shuffle(b_pool)
    rng.shuffle(c_pool)

    # Interleave in strict 4:3:3 pattern: [A, B, C, A, B, C, A, B, C, A] per 10 records
    pattern = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    balanced_train_records = []
    a_i = b_i = c_i = 0
    # 100 updates * 4 microbatches = 400 records (40 cycles)
    for _ in range(40):
        for code in pattern:
            if code == 'A':
                balanced_train_records.append(a_pool[a_i])
                a_i += 1
            elif code == 'B':
                balanced_train_records.append(b_pool[b_i])
                b_i += 1
            elif code == 'C':
                balanced_train_records.append(c_pool[c_i])
                c_i += 1

    assert len(balanced_train_records) == 400
    print(f"  Constructed deterministic balanced 400-record presentation schedule:")
    print(f"    Corpus A: 160 (40.0%) | Corpus B: 120 (30.0%) | Corpus C: 120 (30.0%)")

    train_dataset = MultiTurnDataset(balanced_train_records, tok)
    val_dataset = MultiTurnDataset(val_records, tok)

    # 3. Model Architecture & Parent Weight Initialization
    print("\n[Phase 3] Instantiating Model & Loading Parent Weights...")
    parent_data = torch.load(PARENT_CKPT_PATH, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**parent_data["config"])
    model = ModernGPT(model_cfg)
    model.load_state_dict(parent_data["model_state_dict"])
    print(f"  Loaded state dict from Experiment D Step {parent_data.get('step', 425)}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Model placed on device: {device} ({torch.cuda.get_device_name(0)})")

    # 4. Optimizer, Scheduler & Training Configuration
    print("\n[Phase 4] Initializing Optimizer & Cosine Schedule...")
    total_updates = 100
    warmup_steps = 20
    peak_lr = 1.5e-5
    min_lr = 1.5e-6
    weight_decay = 0.1
    effective_batch_size = 4
    grad_accum_steps = 4

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=get_lr(0, total_updates, warmup_steps, peak_lr, min_lr),
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    training_config = {
        "experiment": "Experiment_F_Stage14_Pilot",
        "parent_checkpoint": str(PARENT_CKPT_PATH),
        "parent_sha256": EXPECTED_PARENT_SHA,
        "tokenizer_sha256": EXPECTED_TOK_SHA,
        "optimizer": "AdamW8bit",
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": weight_decay,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "warmup_updates": warmup_steps,
        "total_updates": total_updates,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": 1,
        "grad_accum_steps": grad_accum_steps,
        "precision": "BF16",
        "grad_clip_norm": 1.0,
        "context_length": 2048,
        "seed": 42,
        "device": torch.cuda.get_device_name(0),
        "timestamp": time.time()
    }

    with open(PILOT_DIR / "training_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(training_config, f, default_flow_style=False)
    print("  Saved training_config.yaml.")

    # Log files
    train_log_file = PILOT_DIR / "training_log.jsonl"
    val_log_file = PILOT_DIR / "validation_metrics.jsonl"
    corpus_loss_file = PILOT_DIR / "corpus_loss_history.json"
    mturn_val_file = PILOT_DIR / "multiturn_validation_results.jsonl"
    cal_val_file = PILOT_DIR / "calibration_validation_results.json"
    syn_val_file = PILOT_DIR / "synthesis_validation_results.json"

    # Step 0 Baseline Evaluation
    print("\n[Phase 5] Running Baseline Evaluation at Step 0...")
    t0_eval = time.time()
    vloss_0 = compute_validation_loss(model, val_dataset, device)
    mturn_m_0, cal_m_0, syn_m_0, mturn_traces_0 = evaluate_autonomous_capabilities(model, tok, val_records, device)
    eval_duration_0 = round(time.time() - t0_eval, 2)
    telemetry_0 = get_gpu_telemetry()

    print(f"  Step 0 Val Loss: {vloss_0['val_loss']} | Perplexity: {vloss_0['perplexity']}")
    print(f"  Multi-turn Trajectory Success: {mturn_m_0['full_trajectory_success_rate']*100:.1f}% ({mturn_m_0['full_trajectory_successes']}/{mturn_m_0['standard_multiturn_total']})")
    print(f"  Calibration F1: {cal_m_0['intent_f1']:.4f} | Concept FPR: {cal_m_0['direct_concept_fpr']*100:.1f}%")
    print(f"  Synthesis Precision: {syn_m_0['strict_supported_precision']*100:.1f}% | Citations: {syn_m_0['citation_correctness']*100:.1f}%")

    val_entry_0 = {
        "step": 0,
        "validation_loss": vloss_0["val_loss"],
        "perplexity": vloss_0["perplexity"],
        "total_target_tokens": vloss_0["total_target_tokens"],
        "per_corpus_loss": vloss_0["per_corpus_loss"],
        "per_category_loss": vloss_0["per_category_loss"],
        "multiturn_metrics": mturn_m_0,
        "calibration_metrics": {k: v for k, v in cal_m_0.items() if k != "cal_results"},
        "synthesis_metrics": {k: v for k, v in syn_m_0.items() if k != "syn_results"},
        "eval_duration_sec": eval_duration_0,
        "telemetry": telemetry_0
    }

    with open(val_log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(val_entry_0) + "\n")

    corpus_loss_history = {
        "steps": [0],
        "overall_val_loss": [vloss_0["val_loss"]],
        "corpus_multiturn": [vloss_0["per_corpus_loss"].get("corpus_multiturn", 0)],
        "corpus_calibration": [vloss_0["per_corpus_loss"].get("corpus_calibration", 0)],
        "corpus_synthesis": [vloss_0["per_corpus_loss"].get("corpus_synthesis", 0)]
    }

    calibration_history = {"step_0": cal_m_0}
    synthesis_history = {"step_0": syn_m_0}

    with open(mturn_val_file, "w", encoding="utf-8") as f:
        for tr in mturn_traces_0:
            f.write(json.dumps({"step": 0, **tr}) + "\n")

    best_val_loss = vloss_0["val_loss"]
    best_step = 0
    best_ckpt_path = CKPT_DIR / "ckpt_pilot_best.pt"

    # Save Step 0 checkpoint as initial best
    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "parent_sha256": EXPECTED_PARENT_SHA,
        "config": model_cfg.__dict__
    }, best_ckpt_path)

    # 5. Training Loop: Steps 1 to 100
    print("\n[Phase 6] Commencing 100-Update Pilot Training Loop...")
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_single)
    train_iter = iter(train_loader)

    current_update = 0
    records_presented = 0
    accum_loss = 0.0
    accum_target_tokens = 0
    total_tokens_seen = 0
    total_assistant_tokens_seen = 0
    sampled_ids = []
    sampled_corpora_counts = {"corpus_multiturn": 0, "corpus_calibration": 0, "corpus_synthesis": 0}

    eval_intervals = [25, 50, 75, 100]
    no_improvement_evals = 0
    t_start_training = time.time()

    model.train()
    optimizer.zero_grad()

    while current_update < total_updates:
        # Microbatch gradient accumulation
        for micro_idx in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            records_presented += 1
            sampled_ids.append(batch["id"])
            sampled_corpora_counts[batch["corpus"]] += 1
            total_tokens_seen += batch["total_tokens"]
            total_assistant_tokens_seen += batch["assistant_tokens"]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = lab[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_scaled = loss / grad_accum_steps

            loss_scaled.backward()
            accum_loss += loss.item()
            accum_target_tokens += (shift_labels != -100).sum().item()

        # Optimizer step
        current_update += 1
        lr_now = get_lr(current_update - 1, total_updates, warmup_steps, peak_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_step_loss = round(accum_loss / grad_accum_steps, 4)
        accum_loss = 0.0
        accum_target_tokens = 0

        # Log training step
        step_log = {
            "update_step": current_update,
            "loss": avg_step_loss,
            "learning_rate": lr_now,
            "grad_norm": round(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm), 4),
            "records_presented": records_presented,
            "total_tokens_seen": total_tokens_seen,
            "assistant_tokens_seen": total_assistant_tokens_seen,
            "effective_epoch": round(records_presented / 800.0, 4),
            "timestamp": time.time()
        }
        with open(train_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_log) + "\n")

        if current_update % 10 == 0 or current_update in eval_intervals:
            print(f"  Step {current_update:03d}/100 | Loss: {avg_step_loss:.4f} | LR: {lr_now:.2e} | GradNorm: {step_log['grad_norm']} | Records: {records_presented}")

        # Scheduled evaluation
        if current_update in eval_intervals:
            print(f"\n  >>> Evaluating Scheduled Checkpoint at Step {current_update}...")
            t_eval_start = time.time()
            vloss = compute_validation_loss(model, val_dataset, device)
            mturn_m, cal_m, syn_m, mturn_traces = evaluate_autonomous_capabilities(model, tok, val_records, device)
            eval_dur = round(time.time() - t_eval_start, 2)
            telemetry = get_gpu_telemetry()

            print(f"  [Step {current_update}] Val Loss: {vloss['val_loss']} | Perplexity: {vloss['perplexity']}")
            print(f"  Multi-turn Trajectory Success: {mturn_m['full_trajectory_success_rate']*100:.1f}% ({mturn_m['full_trajectory_successes']}/{mturn_m['standard_multiturn_total']})")
            print(f"  Calibration F1: {cal_m['intent_f1']:.4f} | Concept FPR: {cal_m['direct_concept_fpr']*100:.1f}%")
            print(f"  Synthesis Precision: {syn_m['strict_supported_precision']*100:.1f}% | Citations: {syn_m['citation_correctness']*100:.1f}%")

            val_entry = {
                "step": current_update,
                "validation_loss": vloss["val_loss"],
                "perplexity": vloss["perplexity"],
                "total_target_tokens": vloss["total_target_tokens"],
                "per_corpus_loss": vloss["per_corpus_loss"],
                "per_category_loss": vloss["per_category_loss"],
                "multiturn_metrics": mturn_m,
                "calibration_metrics": {k: v for k, v in cal_m.items() if k != "cal_results"},
                "synthesis_metrics": {k: v for k, v in syn_m.items() if k != "syn_results"},
                "eval_duration_sec": eval_dur,
                "telemetry": telemetry
            }

            with open(val_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_entry) + "\n")

            corpus_loss_history["steps"].append(current_update)
            corpus_loss_history["overall_val_loss"].append(vloss["val_loss"])
            corpus_loss_history["corpus_multiturn"].append(vloss["per_corpus_loss"].get("corpus_multiturn", 0))
            corpus_loss_history["corpus_calibration"].append(vloss["per_corpus_loss"].get("corpus_calibration", 0))
            corpus_loss_history["corpus_synthesis"].append(vloss["per_corpus_loss"].get("corpus_synthesis", 0))

            calibration_history[f"step_{current_update}"] = cal_m
            synthesis_history[f"step_{current_update}"] = syn_m

            with open(mturn_val_file, "a", encoding="utf-8") as f:
                for tr in mturn_traces:
                    f.write(json.dumps({"step": current_update, **tr}) + "\n")

            # Early stopping and degradation safeguard
            min_delta = 0.001
            if (best_val_loss - vloss["val_loss"]) > min_delta:
                best_val_loss = vloss["val_loss"]
                best_step = current_update
                no_improvement_evals = 0
                print(f"  *** Meaningful Improvement Detected (loss={best_val_loss}) at Step {best_step}! Saving ckpt_pilot_best.pt...")
                torch.save({
                    "step": best_step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "parent_sha256": EXPECTED_PARENT_SHA,
                    "config": model_cfg.__dict__
                }, best_ckpt_path)
            else:
                no_improvement_evals += 1
                print(f"  [Safeguard] No meaningful improvement ({best_val_loss:.4f} vs {vloss['val_loss']:.4f}). Consecutive count: {no_improvement_evals}/3")
                if no_improvement_evals >= 3:
                    print("  [Safeguard Triggered] Early stopping: 3 consecutive evaluations without meaningful improvement.")
                    break

            # Thermal and stability safeguard
            if telemetry.get("temp_c", 0) >= 82.0:
                print(f"  [Thermal Warning] GPU temperature at {telemetry['temp_c']}C. Pausing for 30s cooldown...")
                time.sleep(30)
            if math.isnan(vloss["val_loss"]) or math.isinf(vloss["val_loss"]):
                print("  [Numerical Instability] Validation loss is NaN/Inf! Aborting training.")
                break

            torch.cuda.empty_cache()
            gc.collect()

        if current_update >= total_updates:
            print("  [Mandatory Stop] Reached authorized Step 100 pilot boundary.")
            break

    # Save Step 100 terminal checkpoint
    terminal_ckpt_path = CKPT_DIR / "ckpt_pilot_step100.pt"
    print(f"\n[Phase 7] Saving Terminal Step 100 Checkpoint: {terminal_ckpt_path.name}...")
    torch.save({
        "step": 100,
        "model_state_dict": model.state_dict(),
        "val_loss": vloss["val_loss"],
        "parent_sha256": EXPECTED_PARENT_SHA,
        "config": model_cfg.__dict__
    }, terminal_ckpt_path)

    # Write histories
    with open(corpus_loss_file, "w", encoding="utf-8") as f:
        json.dump(corpus_loss_history, f, indent=2)
    with open(cal_val_file, "w", encoding="utf-8") as f:
        json.dump(calibration_history, f, indent=2)
    with open(syn_val_file, "w", encoding="utf-8") as f:
        json.dump(synthesis_history, f, indent=2)

    # Manifest
    best_sha = sha256_file(best_ckpt_path)
    term_sha = sha256_file(terminal_ckpt_path)

    manifest = {
        "experiment": "QORIN_Experiment_F_Stage14_Pilot",
        "created_at": time.time(),
        "total_updates_completed": 100,
        "records_presented": records_presented,
        "unique_records_sampled": len(set(sampled_ids)),
        "sampled_distribution": {
            "corpus_multiturn": sampled_corpora_counts["corpus_multiturn"],
            "corpus_calibration": sampled_corpora_counts["corpus_calibration"],
            "corpus_synthesis": sampled_corpora_counts["corpus_synthesis"]
        },
        "best_checkpoint": {
            "filename": "ckpt_pilot_best.pt",
            "step": best_step,
            "val_loss": best_val_loss,
            "sha256": best_sha
        },
        "terminal_checkpoint": {
            "filename": "ckpt_pilot_step100.pt",
            "step": 100,
            "val_loss": vloss["val_loss"],
            "sha256": term_sha
        },
        "parent_checkpoint": {
            "path": str(PARENT_CKPT_PATH),
            "step": 425,
            "sha256": EXPECTED_PARENT_SHA
        }
    }

    with open(PILOT_DIR / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_training_time = round(time.time() - t_start_training, 2)
    print("\n" + "=" * 70)
    print("STAGE 14 TRAINING PILOT COMPLETED SUCCESSFULLY!")
    print(f"Total Training & Evaluation Duration: {total_training_time} seconds ({total_training_time/60:.1f} minutes)")
    print(f"Best Checkpoint: Step {best_step} (Loss: {best_val_loss}, SHA: {best_sha[:16]}...)")
    print(f"Terminal Checkpoint: Step 100 (Loss: {vloss['val_loss']}, SHA: {term_sha[:16]}...)")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pilot()
