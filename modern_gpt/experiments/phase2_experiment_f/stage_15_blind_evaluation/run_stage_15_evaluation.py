"""
QORIN Experiment F — Stage 15: Blind Capability Evaluation Runner
Evaluates Experiment F Step 100 candidate vs Frozen Experiment D Step 425 baseline
on the untouched Stage 13 test split (100 records).
"""

import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer
import yaml

REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from backend.tools.python_contract import PythonExecutionContract, PythonExecutionResult
from backend.tools.state_machine import MultiTurnStateMachine, HarnessState, ToolCall, ToolObservation, SessionConfig, SessionTrace, ConversationTurn

# File paths
TEST_SPLIT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_13_dataset/test.jsonl"
TRAIN_SPLIT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_13_dataset/train.jsonl"
VAL_SPLIT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_13_dataset/validation.jsonl"

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"

PARENT_CKPT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
EXPECTED_PARENT_SHA = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"

CANDIDATE_CKPT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt"
EXPECTED_CANDIDATE_SHA = "bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4"

OUT_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_f/stage_15_blind_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are QORIN, a calibrated, evidence-grounded AI assistant.
You have access to the following tools:
- `web_search`: Searches verified databases for factual information.
- `python_interpreter`: Executes isolated Python scripts for calculation and data processing.

Use tools only when strictly necessary. When direct knowledge suffices, answer directly. Always ground answers on retrieved evidence with [1], [2] citations."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@torch.inference_mode()
def make_greedy_generator(model, tok, device):
    stop_ids = {5, 9}  # 5: <|im_end|>, 9: <|endoftext|>
    def generator(prompt_str: str) -> str:
        input_ids = tok.encode(prompt_str).ids
        generated = list(input_ids)
        for _ in range(160):
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
    return generator


def run_evaluation():
    print("=" * 70)
    print("QORIN EXPERIMENT F — STAGE 15: BLIND CAPABILITY EVALUATION")
    print("=" * 70)

    # 1. Contamination Audit
    print("\n[Step 1] Running Test Split Contamination Audit...")
    with open(TRAIN_SPLIT_PATH, encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f]
    with open(VAL_SPLIT_PATH, encoding="utf-8") as f:
        val_recs = [json.loads(line) for line in f]
    with open(TEST_SPLIT_PATH, encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    train_ids = set(r["id"] for r in train_recs)
    val_ids = set(r["id"] for r in val_recs)
    test_ids = set(r["id"] for r in test_recs)

    overlap_train = test_ids.intersection(train_ids)
    overlap_val = test_ids.intersection(val_ids)

    contamination_report = {
        "timestamp": time.time(),
        "train_records": len(train_recs),
        "validation_records": len(val_recs),
        "test_records": len(test_recs),
        "test_train_overlap": len(overlap_train),
        "test_val_overlap": len(overlap_val),
        "status": "PASSED_ZERO_CONTAMINATION" if (len(overlap_train) == 0 and len(overlap_val) == 0) else "FAILED_CONTAMINATION_DETECTED"
    }

    with open(OUT_DIR / "contamination_audit.json", "w", encoding="utf-8") as f:
        json.dump(contamination_report, f, indent=2)

    assert contamination_report["status"] == "PASSED_ZERO_CONTAMINATION", "Test contamination detected!"
    print("  --> Contamination Audit PASSED: Zero overlap across all splits.")

    # 2. Checkpoint Verification
    print("\n[Step 2] Verifying Checkpoints & Hashes...")
    parent_sha = sha256_file(PARENT_CKPT_PATH)
    cand_sha = sha256_file(CANDIDATE_CKPT_PATH)
    tok_sha = sha256_file(TOK_PATH)

    assert parent_sha == EXPECTED_PARENT_SHA, f"Parent SHA mismatch! Expected {EXPECTED_PARENT_SHA}, got {parent_sha}"
    assert cand_sha == EXPECTED_CANDIDATE_SHA, f"Candidate SHA mismatch! Expected {EXPECTED_CANDIDATE_SHA}, got {cand_sha}"

    parent_ckpt_data = torch.load(PARENT_CKPT_PATH, map_location="cpu", weights_only=False)
    cand_ckpt_data = torch.load(CANDIDATE_CKPT_PATH, map_location="cpu", weights_only=False)

    parent_step = parent_ckpt_data.get("step", 425)
    cand_step = cand_ckpt_data.get("step", 100)

    print(f"  Parent Checkpoint: Step {parent_step} | SHA: {parent_sha[:16]}... (VERIFIED)")
    print(f"  Candidate Checkpoint: Step {cand_step} | SHA: {cand_sha[:16]}... (VERIFIED)")

    tok = Tokenizer.from_file(str(TOK_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_config = {
        "evaluation_timestamp": time.time(),
        "prompt_template_sha256": sha256_str(SYSTEM_PROMPT),
        "tokenizer_sha256": tok_sha,
        "models": {
            "baseline": {
                "name": "Frozen Experiment D Best",
                "path": str(PARENT_CKPT_PATH),
                "sha256": parent_sha,
                "optimizer_step": parent_step
            },
            "candidate": {
                "name": "Experiment F Step 100 Pilot Best",
                "path": str(CANDIDATE_CKPT_PATH),
                "sha256": cand_sha,
                "optimizer_step": cand_step
            }
        },
        "decoding": {
            "strategy": "greedy",
            "temperature": 0.0,
            "max_new_tokens": 160,
            "precision": "BF16"
        },
        "dataset": {
            "path": str(TEST_SPLIT_PATH),
            "record_count": len(test_recs)
        }
    }

    with open(OUT_DIR / "evaluation_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    print("  --> Saved evaluation_config.yaml.")

    # 3. Model Instantiation & Evaluation Loops
    models_to_eval = [
        ("baseline", PARENT_CKPT_PATH, parent_ckpt_data),
        ("candidate", CANDIDATE_CKPT_PATH, cand_ckpt_data)
    ]

    all_raw_traces = []
    all_tool_executions = []
    all_record_results = []
    model_summaries = {}

    py_contract = PythonExecutionContract(timeout_seconds=4.0)

    for model_key, ckpt_path, ckpt_dict in models_to_eval:
        print(f"\n[Step 3] Evaluating Model: {model_key.upper()} (Step {ckpt_dict.get('step')})...")
        cfg = ModelConfig(**ckpt_dict["config"])
        model = ModernGPT(cfg).to(device)
        model.load_state_dict(ckpt_dict["model_state_dict"])
        model.eval()

        generator = make_greedy_generator(model, tok, device)

        # Metrics accumulators
        mturn_records = [r for r in test_recs if r["corpus"] == "corpus_multiturn"]
        cal_records = [r for r in test_recs if r["corpus"] == "corpus_calibration"]
        syn_records = [r for r in test_recs if r["corpus"] == "corpus_synthesis"]

        # Corpus A Metrics
        mturn_total = len(mturn_records)
        std_mturn_tasks = [r for r in mturn_records if r["category"] == "MULTITURN_STANDARD_WEB_PYTHON"]
        std_count = len(std_mturn_tasks)
        suff_count = sum(1 for r in mturn_records if r["category"] == "MULTITURN_WEB_SUFFICIENT_DIRECT")
        fail_count = sum(1 for r in mturn_records if r["category"] == "MULTITURN_WEB_FAILURE_RECOVERY")
        contra_count = sum(1 for r in mturn_records if r["category"] == "MULTITURN_CONTRADICTORY_SOURCES")

        turn1_web_count = 0
        turn2_python_count = 0
        py_arg_valid_count = 0
        py_extraction_count = 0
        py_exec_success_count = 0
        numerical_correct_count = 0
        final_synthesis_count = 0
        complete_trajectory_count = 0

        # Sub-category trajectory counters
        std_trajectory_success = 0
        suff_trajectory_success = 0
        fail_trajectory_success = 0
        contra_trajectory_success = 0

        for r in mturn_records:
            rid = r["id"]
            cat = r["category"]
            prompt = r["prompt"]
            target = r.get("target_response", "")

            # Parse recorded observation fixtures from test record ChatML
            chatml_turns = re.findall(r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>', r["serialized_chatml"], re.DOTALL)
            recorded_web_obs = ""
            recorded_py_obs = ""
            recorded_py_code = ""

            for h, c in chatml_turns:
                h_str = h.strip()
                if "tool" in h_str and "call_001" in h_str:
                    recorded_web_obs = c.strip()
                elif "tool" in h_str and "call_002" in h_str:
                    recorded_py_obs = c.strip()
                elif "assistant" in h_str and "python_interpreter" in c:
                    m_code = re.search(r'"code"\s*:\s*"(.*?)"', c, re.DOTALL)
                    if m_code:
                        recorded_py_code = m_code.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")

            # Extract numbers from web observation for extraction check
            numbers_in_web = set(re.findall(r'\b\d+(?:\.\d+)?\b', recorded_web_obs))

            # Turn 1: Model generates tool call
            p1 = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            t0_gen = time.time()
            gen_1 = generator(p1)
            t_gen1_dur = time.time() - t0_gen

            has_web_1 = ("web_search" in gen_1 or "web_research" in gen_1) and ("<tool_call" in gen_1 or "{" in gen_1)
            if has_web_1:
                turn1_web_count += 1

            all_raw_traces.append({
                "model": model_key,
                "record_id": rid,
                "turn": 1,
                "prompt": p1,
                "generation": gen_1,
                "duration_sec": t_gen1_dur
            })

            # Turn 2: Provide recorded Web observation fixture
            p2 = f"{p1}{gen_1}<|im_end|>\n<|im_start|>tool name=web_search id=call_001\n{recorded_web_obs}<|im_end|>\n<|im_start|>assistant\n"
            t0_gen2 = time.time()
            gen_2 = generator(p2)
            t_gen2_dur = time.time() - t0_gen2

            has_py_2 = False
            py_code_valid = False
            extracted_numbers_ok = False
            py_exec_ok = False
            actual_py_output = ""
            py_code_generated = ""

            if cat == "MULTITURN_STANDARD_WEB_PYTHON":
                has_py_2 = ("python_interpreter" in gen_2 or "python" in gen_2) and ("<tool_call" in gen_2 or "{" in gen_2)
                if has_py_2:
                    turn2_python_count += 1

                # Parse Python code from gen_2
                m_py = re.search(r'\{.*\}', gen_2, re.DOTALL)
                if m_py:
                    try:
                        args = json.loads(m_py.group(0), strict=False)
                        code = args.get("arguments", {}).get("code", "") if "arguments" in args else args.get("code", "")
                        py_code_generated = code
                        if code and code.strip():
                            ast.parse(code)
                            py_code_valid = True
                            py_arg_valid_count += 1

                            # Value extraction check: did the model use numbers from the Web observation?
                            nums_in_code = set(re.findall(r'\b\d+(?:\.\d+)?\b', code))
                            if len(nums_in_code.intersection(numbers_in_web)) > 0:
                                extracted_numbers_ok = True
                                py_extraction_count += 1

                            # Execute Python code live in contract
                            exec_res: PythonExecutionResult = py_contract.execute(code)
                            all_tool_executions.append({
                                "model": model_key,
                                "record_id": rid,
                                "tool": "python_interpreter",
                                "code": code,
                                "status": exec_res.status,
                                "stdout": exec_res.stdout,
                                "stderr": exec_res.stderr,
                                "duration_ms": getattr(exec_res, "execution_time_ms", 0.0)
                            })
                            if exec_res.status == "SUCCESS":
                                py_exec_ok = True
                                py_exec_success_count += 1
                                actual_py_output = exec_res.stdout.strip()
                    except Exception as e:
                        py_code_valid = False

                all_raw_traces.append({
                    "model": model_key,
                    "record_id": rid,
                    "turn": 2,
                    "prompt": p2,
                    "generation": gen_2,
                    "duration_sec": t_gen2_dur
                })

                # Turn 3: Final Synthesis
                obs_to_feed = actual_py_output if actual_py_output else recorded_py_obs
                p3 = f"{p2}{gen_2}<|im_end|>\n<|im_start|>tool name=python_interpreter id=call_002\n{obs_to_feed}<|im_end|>\n<|im_start|>assistant\n"
                t0_gen3 = time.time()
                gen_3 = generator(p3)
                t_gen3_dur = time.time() - t0_gen3

                all_raw_traces.append({
                    "model": model_key,
                    "record_id": rid,
                    "turn": 3,
                    "prompt": p3,
                    "generation": gen_3,
                    "duration_sec": t_gen3_dur
                })

                # Numerical & Synthesis correctness
                has_citation = "[1]" in gen_3 or "[2]" in gen_3
                # Check target numbers
                target_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', target))
                final_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', gen_3))
                num_match = len(target_nums.intersection(final_nums)) > 0 or (len(actual_py_output) > 0 and actual_py_output in gen_3)

                if num_match:
                    numerical_correct_count += 1
                if has_citation and len(gen_3.strip()) > 25:
                    final_synthesis_count += 1

                traj_ok = has_web_1 and has_py_2 and py_code_valid and py_exec_ok and has_citation
                if traj_ok:
                    complete_trajectory_count += 1
                    std_trajectory_success += 1

                all_record_results.append({
                    "model": model_key,
                    "id": rid,
                    "corpus": "corpus_multiturn",
                    "category": cat,
                    "turn1_web": has_web_1,
                    "turn2_python": has_py_2,
                    "py_code_valid": py_code_valid,
                    "py_extracted_values": extracted_numbers_ok,
                    "py_exec_success": py_exec_ok,
                    "numerical_correct": num_match,
                    "citation_correct": has_citation,
                    "trajectory_success": traj_ok,
                    "final_response": gen_3[:120]
                })

            else:
                # Other multi-turn categories (sufficient direct, failure recovery, contradictory)
                all_raw_traces.append({
                    "model": model_key,
                    "record_id": rid,
                    "turn": 2,
                    "prompt": p2,
                    "generation": gen_2,
                    "duration_sec": t_gen2_dur
                })
                has_cite = "[1]" in gen_2 or "[2]" in gen_2
                non_empty = len(gen_2.strip()) > 30

                traj_other_ok = has_web_1 and non_empty
                if cat == "MULTITURN_WEB_SUFFICIENT_DIRECT" and traj_other_ok:
                    suff_trajectory_success += 1
                elif cat == "MULTITURN_WEB_FAILURE_RECOVERY" and traj_other_ok:
                    fail_trajectory_success += 1
                elif cat == "MULTITURN_CONTRADICTORY_SOURCES" and traj_other_ok:
                    contra_trajectory_success += 1

                all_record_results.append({
                    "model": model_key,
                    "id": rid,
                    "corpus": "corpus_multiturn",
                    "category": cat,
                    "turn1_web": has_web_1,
                    "turn2_python": False,
                    "py_code_valid": False,
                    "py_extracted_values": False,
                    "py_exec_success": False,
                    "numerical_correct": False,
                    "citation_correct": has_cite,
                    "trajectory_success": traj_other_ok,
                    "final_response": gen_2[:120]
                })

        corpus_a_metrics = {
            "first_turn_web_selection_acc": round(turn1_web_count / max(1, mturn_total), 4),
            "first_turn_web_selection_raw": f"{turn1_web_count}/{mturn_total}",
            "second_turn_python_selection_acc": round(turn2_python_count / max(1, std_count), 4),
            "second_turn_python_selection_raw": f"{turn2_python_count}/{std_count}",
            "python_argument_validity_acc": round(py_arg_valid_count / max(1, std_count), 4),
            "python_argument_validity_raw": f"{py_arg_valid_count}/{std_count}",
            "value_extraction_accuracy": round(py_extraction_count / max(1, std_count), 4),
            "value_extraction_raw": f"{py_extraction_count}/{std_count}",
            "python_execution_success_rate": round(py_exec_success_count / max(1, std_count), 4),
            "python_execution_success_raw": f"{py_exec_success_count}/{std_count}",
            "numerical_answer_correctness": round(numerical_correct_count / max(1, std_count), 4),
            "numerical_answer_raw": f"{numerical_correct_count}/{std_count}",
            "final_evidence_synthesis_acc": round(final_synthesis_count / max(1, std_count), 4),
            "final_evidence_synthesis_raw": f"{final_synthesis_count}/{std_count}",
            "complete_end_to_end_trajectory_rate": round(complete_trajectory_count / max(1, std_count), 4),
            "complete_end_to_end_trajectory_raw": f"{complete_trajectory_count}/{std_count}",
            "sub_categories": {
                "standard_web_python_success": f"{std_trajectory_success}/{std_count}",
                "web_sufficient_direct_success": f"{suff_trajectory_success}/{suff_count}",
                "web_failure_recovery_success": f"{fail_trajectory_success}/{fail_count}",
                "contradictory_sources_success": f"{contra_trajectory_success}/{contra_count}"
            }
        }

        # Corpus B: Decision Calibration (30 records)
        direct_concept_fp = 0
        direct_concept_total = 0
        direct_concept_correct = 0

        web_call_tp = 0
        web_call_required = 0

        py_call_tp = 0
        py_call_required = 0

        total_tool_calls_emitted = 0
        unnecessary_tool_calls = 0
        missed_tool_calls = 0

        for r in cal_records:
            rid = r["id"]
            cat = r["category"]
            prompt = r["prompt"]

            p = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            t0 = time.time()
            gen = generator(p)
            dur = time.time() - t0

            all_raw_traces.append({
                "model": model_key,
                "record_id": rid,
                "turn": 1,
                "prompt": p,
                "generation": gen,
                "duration_sec": dur
            })

            has_tool = ("<tool_call" in gen or "{\"name\":" in gen or "web_search" in gen or "python" in gen) and not gen.startswith("The") and not gen.startswith("A")

            if has_tool:
                total_tool_calls_emitted += 1

            correct_decision = False
            if cat == "CAL_DIRECT_CONCEPT":
                direct_concept_total += 1
                if has_tool:
                    direct_concept_fp += 1
                    unnecessary_tool_calls += 1
                else:
                    direct_concept_correct += 1
                    correct_decision = True
            elif cat == "CAL_DIRECT_PROVIDED_CALC":
                py_call_required += 1
                if "python_interpreter" in gen or "python" in gen:
                    py_call_tp += 1
                    correct_decision = True
                else:
                    missed_tool_calls += 1
            elif cat == "CAL_DIRECT_FACTUAL_WEB":
                web_call_required += 1
                if "web_search" in gen or "web_research" in gen:
                    web_call_tp += 1
                    correct_decision = True
                else:
                    missed_tool_calls += 1
            elif cat == "CAL_CONTRASTIVE_PAIRS":
                if "python_interpreter" in r["serialized_chatml"]:
                    py_call_required += 1
                    if "python" in gen:
                        py_call_tp += 1
                        correct_decision = True
                    else:
                        missed_tool_calls += 1
                elif "web_search" in r["serialized_chatml"]:
                    web_call_required += 1
                    if "web" in gen:
                        web_call_tp += 1
                        correct_decision = True
                    else:
                        missed_tool_calls += 1
                else:
                    direct_concept_total += 1
                    if has_tool:
                        direct_concept_fp += 1
                        unnecessary_tool_calls += 1
                    else:
                        direct_concept_correct += 1
                        correct_decision = True

            all_record_results.append({
                "model": model_key,
                "id": rid,
                "corpus": "corpus_calibration",
                "category": cat,
                "has_tool_call": has_tool,
                "correct_decision": correct_decision,
                "generation": gen[:120]
            })

        corpus_b_metrics = {
            "conceptual_direct_fpr": round(direct_concept_fp / max(1, direct_concept_total), 4),
            "conceptual_direct_fpr_raw": f"{direct_concept_fp}/{direct_concept_total}",
            "correct_direct_answer_rate": round(direct_concept_correct / max(1, direct_concept_total), 4),
            "correct_direct_answer_raw": f"{direct_concept_correct}/{direct_concept_total}",
            "required_web_call_recall": round(web_call_tp / max(1, web_call_required), 4),
            "required_web_call_recall_raw": f"{web_call_tp}/{web_call_required}",
            "required_python_call_recall": round(py_call_tp / max(1, py_call_required), 4),
            "required_python_call_recall_raw": f"{py_call_tp}/{py_call_required}",
            "unnecessary_tool_calls_count": unnecessary_tool_calls,
            "missed_tool_calls_count": missed_tool_calls
        }

        # Corpus C: Evidence-Grounded Synthesis (30 records)
        full_support_tasks = [r for r in syn_records if r["category"] == "SYN_FULL_SUPPORT"]
        part_support_tasks = [r for r in syn_records if r["category"] == "SYN_PARTIAL_SUPPORT"]
        contra_support_tasks = [r for r in syn_records if r["category"] == "SYN_CONTRADICTORY_EVIDENCE"]
        insuff_support_tasks = [r for r in syn_records if r["category"] == "SYN_INSUFFICIENT_EVIDENCE"]

        full_support_correct = 0
        citation_correct = 0
        unsupported_claims = 0
        partial_handled = 0
        contradiction_handled = 0
        numerical_extracted = 0
        semantic_success = 0

        for r in syn_records:
            rid = r["id"]
            cat = r["category"]
            prompt = r["prompt"]
            target = r.get("target_response", "")

            # Extract tool observation fixture
            chatml_turns = re.findall(r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>', r["serialized_chatml"], re.DOTALL)
            obs_content = ""
            for h, c in chatml_turns:
                if "tool" in h.strip():
                    obs_content = c.strip()

            p = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>tool name=web_search id=call_001\n{obs_content}<|im_end|>\n<|im_start|>assistant\n"
            t0 = time.time()
            gen = generator(p)
            dur = time.time() - t0

            all_raw_traces.append({
                "model": model_key,
                "record_id": rid,
                "turn": 1,
                "prompt": p,
                "generation": gen,
                "duration_sec": dur
            })

            gen_lower = gen.lower()
            has_cite = "[1]" in gen or "[2]" in gen

            # Numbers in observation vs response
            nums_obs = set(re.findall(r'\b\d+(?:\.\d+)?\b', obs_content))
            nums_gen = set(re.findall(r'\b\d+(?:\.\d+)?\b', gen))
            num_extracted_ok = len(nums_gen.intersection(nums_obs)) > 0

            is_semantic_ok = False
            if cat == "SYN_FULL_SUPPORT":
                if has_cite:
                    citation_correct += 1
                if num_extracted_ok:
                    numerical_extracted += 1
                # Support check: grounded in text
                if len(gen.strip()) > 30 and (has_cite or num_extracted_ok):
                    full_support_correct += 1
                    is_semantic_ok = True
                else:
                    unsupported_claims += 1

            elif cat == "SYN_PARTIAL_SUPPORT":
                if any(w in gen_lower for w in ["partial", "approximate", "estimated", "preliminary", "however", "while", "incomplete"]):
                    partial_handled += 1
                    is_semantic_ok = True

            elif cat == "SYN_CONTRADICTORY_EVIDENCE":
                if any(w in gen_lower for w in ["conflict", "discrepanc", "differ", "vary", "range", "between", "alternative", "diverge"]):
                    contradiction_handled += 1
                    is_semantic_ok = True

            elif cat == "SYN_INSUFFICIENT_EVIDENCE":
                if any(w in gen_lower for w in ["insufficient", "not provided", "does not contain", "no record", "unknown", "unavailable"]):
                    is_semantic_ok = True

            if is_semantic_ok:
                semantic_success += 1

            all_record_results.append({
                "model": model_key,
                "id": rid,
                "corpus": "corpus_synthesis",
                "category": cat,
                "has_citation": has_cite,
                "numerical_extracted": num_extracted_ok,
                "semantic_success": is_semantic_ok,
                "generation": gen[:120]
            })

        corpus_c_metrics = {
            "claim_level_supported_precision": round(full_support_correct / max(1, len(full_support_tasks)), 4),
            "claim_level_supported_raw": f"{full_support_correct}/{len(full_support_tasks)}",
            "citation_correctness_rate": round(citation_correct / max(1, len(full_support_tasks)), 4),
            "citation_correctness_raw": f"{citation_correct}/{len(full_support_tasks)}",
            "unsupported_claim_rate": round(unsupported_claims / max(1, len(full_support_tasks)), 4),
            "unsupported_claim_raw": f"{unsupported_claims}/{len(full_support_tasks)}",
            "partial_support_handling_rate": round(partial_handled / max(1, len(part_support_tasks)), 4),
            "partial_support_raw": f"{partial_handled}/{len(part_support_tasks)}",
            "contradiction_handling_rate": round(contradiction_handled / max(1, len(contra_support_tasks)), 4),
            "contradiction_handling_raw": f"{contradiction_handled}/{len(contra_support_tasks)}",
            "numerical_extraction_accuracy": round(numerical_extracted / max(1, len(full_support_tasks)), 4),
            "numerical_extraction_raw": f"{numerical_extracted}/{len(full_support_tasks)}",
            "final_answer_semantic_success_rate": round(semantic_success / max(1, len(syn_records)), 4),
            "final_answer_semantic_success_raw": f"{semantic_success}/{len(syn_records)}"
        }

        model_summaries[model_key] = {
            "model_name": model_key,
            "step": ckpt_dict.get("step"),
            "corpus_a_multiturn": corpus_a_metrics,
            "corpus_b_calibration": corpus_b_metrics,
            "corpus_c_synthesis": corpus_c_metrics
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Live Web Smoke Test (3 distinct tasks)
    print("\n[Step 4] Running Live Web Smoke Test...")
    smoke_tasks = [
        {"id": "smoke_001", "prompt": "What is the equatorial radius of planet Saturn in kilometers?"},
        {"id": "smoke_002", "prompt": "What is the speed of sound in dry air at 20 degrees Celsius in m/s?"},
        {"id": "smoke_003", "prompt": "Who won the FIFA Men's World Cup in 2022?"}
    ]

    smoke_results = []
    # Test candidate on live smoke test
    cfg = ModelConfig(**cand_ckpt_data["config"])
    cand_model = ModernGPT(cfg).to(device)
    cand_model.load_state_dict(cand_ckpt_data["model_state_dict"])
    cand_model.eval()
    cand_gen = make_greedy_generator(cand_model, tok, device)

    for st in smoke_tasks:
        p = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{st['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        gen = cand_gen(p)
        has_tool = "web_search" in gen or "web_research" in gen
        smoke_results.append({
            "id": st["id"],
            "prompt": st["prompt"],
            "generation": gen[:120],
            "emitted_web_call": has_tool
        })

    del cand_model
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Write Artifacts
    print("\n[Step 5] Writing Deliverables...")

    with open(OUT_DIR / "raw_generation_traces.jsonl", "w", encoding="utf-8") as f:
        for t in all_raw_traces:
            f.write(json.dumps(t) + "\n")

    with open(OUT_DIR / "tool_execution_traces.jsonl", "w", encoding="utf-8") as f:
        for ex in all_tool_executions:
            f.write(json.dumps(ex) + "\n")

    with open(OUT_DIR / "per_record_results.jsonl", "w", encoding="utf-8") as f:
        for res in all_record_results:
            f.write(json.dumps(res) + "\n")

    with open(OUT_DIR / "per_corpus_metrics.json", "w", encoding="utf-8") as f:
        json.dump(model_summaries, f, indent=2)

    comparison = {
        "timestamp": time.time(),
        "baseline": model_summaries["baseline"],
        "candidate": model_summaries["candidate"],
        "smoke_test": smoke_results
    }
    with open(OUT_DIR / "parent_vs_candidate_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("Stage 15 Evaluation Data Processing COMPLETE.")


if __name__ == "__main__":
    run_evaluation()
